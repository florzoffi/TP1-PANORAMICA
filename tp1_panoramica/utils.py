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

# 1) helper: de coords -> KeyPoints (sin re-detectar)
def kps_from_coords(coords, size=7):
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in coords]

# 2) describir SOLO los puntos ANMS
def describe_orb_on_points(img_bgr, coords_anms, nfeatures=2000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    kps = kps_from_coords(coords_anms)
    kps2, desc = orb.compute(gray, kps)  # compute (no detect)
    return kps2 if kps2 is not None else [], desc

# 3) matching ORB (NORM_HAMMING) con Lowe + cross-check opcional
def match_orb(desc1, desc2, ratio=0.75, crosscheck=True):
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = [m for m,n in m12 if n is not None and m.distance < ratio*n.distance]
    if not crosscheck:
        return good12
    m21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = [m for m,n in m21 if n is not None and m.distance < ratio*n.distance]
    mutual = {(m.queryIdx, m.trainIdx) for m in good21}
    return [m for m in good12 if (m.trainIdx, m.queryIdx) in mutual]

# 4) dibujar matches
def draw_matches(img1, kps1, img2, kps2, matches, max_lines=80, title="Matches"):
    vis = cv2.drawMatches(img1, kps1, img2, kps2, matches[:max_lines], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12,6)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (shown={min(len(matches),max_lines)} / total={len(matches)})")
    plt.axis("off"); plt.show()