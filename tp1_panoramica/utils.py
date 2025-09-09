import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_corners(img, method='harris'):
    useHarrisDetector = method == 'harris'
    img = np.float32(img)
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=1000,
        qualityLevel=0.05,
        minDistance=11,
        useHarrisDetector=useHarrisDetector
    )
    return corners.reshape(-1, 2)

def plot_corners(img, method='shi-tomasi', maxCorners=1000, qualityLevel=0.05, minDistance=11):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        useHarrisDetector=(method == 'harris')
    )

    corners = corners.reshape(-1, 2)

    plt.imshow(gray, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], s=50, marker='+', color='red')
    plt.title(f"Detección de esquinas ({method})")
    plt.axis("off")
    plt.show()

def anms(keypoints, descriptors, N=500, strength_ratio=1.0, eps=1e-12):
    """
    Adaptive Non-Maximal Suppression.
    - keypoints: lista de cv2.KeyPoint
    - descriptors: np.ndarray alineado con keypoints
    - N: cantidad final deseada
    - strength_ratio: cuán 'más fuerte' debe ser el punto que suprime (>=1.0)
    - Devuelve: keypoints_filtrados, descriptors_filtrados
    """

    if len(keypoints) <= N:
        return keypoints, descriptors

    # Tomo (x, y, response) y ordeno por respuesta (desc)
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # (n,2)
    resp = np.array([kp.response for kp in keypoints], dtype=np.float32)
    order = np.argsort(-resp)  # índices de mayor a menor
    pts_sorted = pts[order]
    resp_sorted = resp[order]
    kps_sorted = [keypoints[i] for i in order]
    desc_sorted = descriptors[order] if descriptors is not None else None

    n = len(kps_sorted)
    # Radio de supresión para cada punto
    # Para el más fuerte: infinito (no lo suprime nadie)
    radii = np.full(n, np.inf, dtype=np.float32)

    # Para cada punto (de fuerte a débil), busco la mínima distancia
    # al vecino más cercano que tenga respuesta > strength_ratio * resp_i
    for i in range(1, n):  # i=0 es el más fuerte
        # candidatos que pueden suprimir a i: j < i (tienen resp >= resp_i)
        # aplico ratio (>= strength_ratio * resp_i)
        mask_stronger = resp_sorted[:i] > (strength_ratio * resp_sorted[i] + eps)
        if not np.any(mask_stronger):
            # si no hay claramente más fuertes, igual uso todos los anteriores
            mask_stronger = np.ones(i, dtype=bool)

        dif = pts_sorted[:i][mask_stronger] - pts_sorted[i]
        d2 = np.sum(dif * dif, axis=1)
        radii[i] = np.sqrt(np.min(d2))

    # Tomo los N con mayor radio (más aislados/representativos)
    keep_idx_sorted = np.argsort(-radii)[:N]
    keep_idx = order[keep_idx_sorted]  # volver a índices originales

    kps_out = [keypoints[i] for i in keep_idx]
    if descriptors is not None:
        desc_out = descriptors[keep_idx]
    else:
        desc_out = None

    return kps_out, desc_out

def to_gray_clahe(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(g)

# --- util: máscara para ignorar cielo (recorta el 35% superior) ---
def sky_mask(img, top_frac=0.35):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(h*top_frac):,:] = 255
    return m

# --- util: Lowe ratio simétrico (kNN en ambos sentidos) ---
def symmetric_lowe(desc1, desc2, ratio=0.8):  # subir a 0.8 ayuda en escenas repetitivas
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = []
    for m,n in knn12:
        if m.distance < ratio * n.distance:
            good12.append((m.queryIdx, m.trainIdx, m))
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for m,n in knn21:
        if m.distance < ratio * n.distance:
            good21.append((m.queryIdx, m.trainIdx))
    set21 = set((q,t) for q,t in good21)  # (idx_desc2, idx_desc1)
    sym = [m for q,t,m in good12 if (t,q) in set21]
    return sym

# --- util: RANSAC para medir inliers y depurar geométricamente ---
def ransac_inliers(kp1, kp2, matches, reproj_thresh=4.0):
    if len(matches) < 4:
        return None, [], []
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return None, [], []
    mask = mask.ravel().astype(bool)
    inliers = [m for m,ok in zip(matches, mask) if ok]
    outliers = [m for m,ok in zip(matches, mask) if not ok]
    return H, inliers, outliers

# --- util: dibujar matches ---
def draw_matches(img1, kp1, img2, kp2, matches, title="", max_display=80):
    matches = sorted(matches, key=lambda m: m.distance)[:max_display]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title); plt.axis("off"); plt.show()