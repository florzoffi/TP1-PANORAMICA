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