import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_corners(img, method='harris',
                 maxCorners=1000, qualityLevel=0.05, minDistance=11,
                 blockSize=3, k_harris=0.04):
    """
    Devuelve:
      - coords: (N,2) en (x,y)
      - responses: (N,) respuesta de esquina (Harris o Shi-Tomasi)
      - keypoints: lista de cv2.KeyPoint con .response poblado
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    useHarris = (method == 'harris')
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=maxCorners, qualityLevel=qualityLevel,
        minDistance=minDistance, blockSize=blockSize,
        useHarrisDetector=useHarris, k=(k_harris if useHarris else 0.0)
    )
    if pts is None:
        return np.empty((0,2), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    coords = pts.reshape(-1, 2)

    # Mapa de respuesta para poblar .response
    if useHarris:
        respmap = cv2.cornerHarris(gray, blockSize=blockSize, ksize=3, k=k_harris)
    else:
        respmap = cv2.cornerMinEigenVal(gray, blockSize=blockSize, ksize=3)

    H, W = gray.shape
    responses = []
    keypoints = []
    for x, y in coords:
        xi = int(np.clip(round(x), 0, W-1))
        yi = int(np.clip(round(y), 0, H-1))
        r = float(respmap[yi, xi])
        responses.append(r)
        keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=7, response=r))

    return coords.astype(np.float32), np.array(responses, dtype=np.float64), keypoints

def plot_corners(img, method='shi-tomasi', maxCorners=1000, qualityLevel=0.05, minDistance=11):
    """
    Igual que la tuya, pero internamente usa find_corners() y
    devuelve coords/responses/kps por si querés guardarlos.
    """
    coords, responses, _ = find_corners(
        img, method=method, maxCorners=maxCorners,
        qualityLevel=qualityLevel, minDistance=minDistance
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    if len(coords) > 0:
        plt.scatter(coords[:, 0], coords[:, 1], s=50, marker='+', color='red')
    plt.title(f"Detección de esquinas ({method})  n={len(coords)}")
    plt.axis("off")
    plt.show()
    return coords, responses


def anms_from_coords(coords, responses, N=600, strength_ratio=1.1):
    """
    Radio de supresión respecto de vecinos con r_j > c * r_i.
    Si un punto no tiene vecino más fuerte, en lugar de inf usamos el
    máximo de D^2 para que sea comparable/finito.
    """
    n = len(coords)
    if n == 0:
        return np.array([], dtype=int), np.array([])

    xy = coords.astype(np.float64)
    r  = responses.astype(np.float64)

    dx = xy[:,0][:,None] - xy[:,0][None,:]
    dy = xy[:,1][:,None] - xy[:,1][None,:]
    D2 = dx*dx + dy*dy
    np.fill_diagonal(D2, np.inf)

    # vecinos "más fuertes" con un margen (c > 1)
    stronger = r[None,:] > (strength_ratio * r[:,None])
    D2_masked = np.where(stronger, D2, np.inf)

    R = D2_masked.min(axis=1)
    # donde no hubo vecino más fuerte, ponemos un radio grande pero FINITO
    no_stronger = ~np.isfinite(R)
    if np.any(no_stronger):
        R[no_stronger] = np.nanmax(np.where(np.isfinite(D2), D2, 0.0))

    keep = np.argsort(-R)[:min(N, n)]
    return keep, R[keep]

def show_points_overlay(img_bgr, pts_xy, title):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    rad = max(2, int(0.004 * min(H, W)))      # radio relativo al tamaño de imagen
    thick = max(1, int(rad // 2))
    for x, y in pts_xy.astype(int):
        cv2.circle(vis, (x, y), rad, (0, 255, 0), thickness=thick, lineType=cv2.LINE_AA)
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (n={len(pts_xy)})"); plt.axis("off"); plt.show()


def to_gray_clahe(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def kps_from_coords(coords, size=7):
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in coords]

def build_orb():
    # más robusto a leves escalas/rotaciones y poco contraste
    return cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.1,
        nlevels=12,
        edgeThreshold=15,
        patchSize=31,
        WTA_K=2,
        fastThreshold=5
    )

def describe_orb_on_points(img, coords_anms, orb=None):
    if coords_anms is None or len(coords_anms) == 0:
        return [], None
    if orb is None:
        orb = build_orb()
    gray = to_gray_clahe(img)
    kps = kps_from_coords(coords_anms)
    kps2, desc = orb.compute(gray, kps)
    return (kps2 if kps2 is not None else []), desc

def match_orb(desc1, desc2, policy="ratio", ratio=0.85):
    """
    policy: "ratio" | "cross" | "both"
    """
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return []
    if policy == "cross":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return sorted(bf.match(desc1, desc2), key=lambda m: m.distance)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # ratio adelante
    m12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = [m for m,n in m12 if n is not None and m.distance < ratio*n.distance]
    if policy == "ratio":
        return good12

    # both (ratio + cross-check)
    m21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = [m for m,n in m21 if n is not None and m.distance < ratio*n.distance]
    mutual = {(m.queryIdx, m.trainIdx) for m in good21}
    return [m for m in good12 if (m.trainIdx, m.queryIdx) in mutual]

def draw_matches(img1, kps1, img2, kps2, matches,
                       max_lines=200, thickness=4, radius=6,
                       color=None, title="Matches (thick)"):
    """
    Dibuja matches con grosor configurable.
    - thickness: grosor de las líneas
    - radius: radio de los círculos en los puntos
    - color: (B,G,R). Si None, colorea cada línea distinto.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    rng = np.random.default_rng(123)

    shown = min(len(matches), max_lines)
    for i in range(shown):
        m = matches[i]
        x1, y1 = map(int, np.round(kps1[m.queryIdx].pt))
        x2, y2 = map(int, np.round(kps2[m.trainIdx].pt))
        p1 = (x1, y1)
        p2 = (x2 + w1, y2)  # offset a la derecha

        # color por defecto: aleatorio (pero reproducible)
        c = tuple(int(v) for v in (rng.integers(64, 255, size=3) if color is None else color))
        # círculos y línea gorditos
        cv2.circle(canvas, p1, radius, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, c, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1, p2, c, thickness, lineType=cv2.LINE_AA)

    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (shown={shown} / total={len(matches)})")
    plt.axis("off")
    plt.show()