import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------- 3.1 Harris, Shi-Tomasi ----------
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

# ---------- 3.2 Harris, Shi-Tomasi ----------
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

    stronger = r[None,:] > (strength_ratio * r[:,None])
    D2_masked = np.where(stronger, D2, np.inf)

    R = D2_masked.min(axis=1)
    no_stronger = ~np.isfinite(R)
    if np.any(no_stronger):
        R[no_stronger] = np.nanmax(np.where(np.isfinite(D2), D2, 0.0))

    keep = np.argsort(-R)[:min(N, n)]
    return keep, R[keep]

def show_points_overlay(img_bgr, pts_xy, title):
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    rad = max(2, int(0.004 * min(H, W)))      
    thick = max(1, int(rad // 2))
    for x, y in pts_xy.astype(int):
        cv2.circle(vis, (x, y), rad, (0, 255, 0), thickness=thick, lineType=cv2.LINE_AA)
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (n={len(pts_xy)})"); plt.axis("off"); plt.show()

def kps_from_coords(coords, size=7):
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in coords]

def describe_orb_on_points(img_bgr, coords_anms, nfeatures=2000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7)
    kps = kps_from_coords(coords_anms)
    kps2, desc = orb.compute(gray, kps) 
    return kps2 if kps2 is not None else [], desc

# ---------- 3.1/3.2 SIFT ----------
def anms_from_keypoints(keypoints, descriptors, N=800, strength_ratio=1.0, eps=1e-12):
    if len(keypoints) <= N:
        return keypoints, descriptors
    pts   = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    resp  = np.array([kp.response for kp in keypoints], dtype=np.float32)
    order = np.argsort(-resp)
    pts_s, resp_s = pts[order], resp[order]
    kps_s = [keypoints[i] for i in order]
    desc_s = descriptors[order] if descriptors is not None else None

    n = len(kps_s)
    radii = np.full(n, np.inf, np.float32)
    for i in range(1, n):
        stronger = resp_s[:i] > (strength_ratio * resp_s[i] + eps)
        if not np.any(stronger):
            stronger = np.ones(i, dtype=bool)
        dif = pts_s[:i][stronger] - pts_s[i]
        d2 = np.sum(dif*dif, axis=1)
        radii[i] = np.sqrt(np.min(d2))
    keep_sorted = np.argsort(-radii)[:N]
    keep_idx = order[keep_sorted]
    kps_out = [keypoints[i] for i in keep_idx]
    desc_out = descriptors[keep_idx] if descriptors is not None else None
    return kps_out, desc_out

def topN_keypoints(keypoints, descriptors, N=500):
    if len(keypoints) <= N: 
        return keypoints, descriptors
    idx = np.argsort([-kp.response for kp in keypoints])[:N]
    kps_out = [keypoints[i] for i in idx]
    desc_out = descriptors[idx] if descriptors is not None else None
    return kps_out, desc_out

def detect_sift(img, nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, mask=None):
    """
    Devuelve:
      - coords: (N,2) en (x,y)
      - responses: (N,)
      - keypoints: lista cv2.KeyPoint
      - descriptors: (N,128) float32
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold, sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if keypoints is None or len(keypoints) == 0:
        return (np.empty((0,2), np.float32), np.empty((0,), np.float32), [], None)
    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    return coords, responses, keypoints, descriptors

def plot_sift_points(img, keypoints, title="SIFT keypoints", color=(255,0,0)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    vis = cv2.drawKeypoints(
        gray_bgr, keypoints, None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()

def run_anms_sift(corners_storeSF, keep_frac=0.30, min_keep=300, max_keep=8000, strength_ratio=1.1, show=True):
    """
    corners_storeSF: dict con {"img","kps","desc"} (tu formato SIFT)
    Devuelve: dos dicts con KPs y Descs ya filtrados por ANMS.
    ANMS es adaptativo por imagen: combina porcentaje + [min,max].
    """
    kps_anms_dict  = {}
    desc_anms_dict = {}

    for path, data in corners_storeSF.items():
        img  = data["img"]
        kps  = data["kps"]
        desc = data["desc"]

        n0 = len(kps)
        if n0 == 0:
            print(f"[{path}] SIFT: sin keypoints.")
            kps_anms_dict[path], desc_anms_dict[path] = [], None
            continue

        # N adaptativo por imagen
        N_target = max(min_keep, min(int(keep_frac * n0), max_keep))

        # ANMS específico para SIFT (filtra KPs y alinea Descs)
        kps_anms, desc_anms = anms_from_keypoints(
            kps, desc, N=N_target, strength_ratio=strength_ratio
        )

        print(f"[{path}] SIFT antes={n0}  después(ANMS)={len(kps_anms)}  "
              f"(N_target={N_target}, c={strength_ratio})  "
              f"desc={None if desc_anms is None else desc_anms.shape}")

        if show:
            plot_sift_points(img, kps_anms, title=f"SIFT-ANMS {len(kps_anms)} - {path}", color=(255,0,0))

        kps_anms_dict[path]  = kps_anms
        desc_anms_dict[path] = desc_anms

    return kps_anms_dict, desc_anms_dict