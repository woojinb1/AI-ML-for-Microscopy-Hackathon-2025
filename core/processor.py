import numpy as np
import cv2
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# [추가] 고급 분석 라이브러리
from scipy.ndimage import gaussian_filter, gaussian_laplace, gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import alphashape
    ALPHASHAPE_AVAILABLE = True
except ImportError:
    ALPHASHAPE_AVAILABLE = False

# =========================================================
# [수정됨] GMM + RDF 기반 고급 Peak 탐지 (순서 교정)
# =========================================================
def detect_peaks_advanced(fft_log, params):
    """
    순수한 FFT Log Magnitude를 받아서 내부적으로 통계 처리 후 Peak 반환
    """
    # 1. 파라미터 로드
    bg_sigma = 20
    rd_sigma = 3
    rd_prominence = 0.02
    rd_distance = 10
    bw = 20  # Ring bandwidth
    
    log_sigma = 2.0
    min_dist = 4
    
    # LoG threshold: 너무 낮으면 노이즈가 너무 많이 후보가 됨. 적당히 낮게 설정.
    threshold_rel = 0.5 
    
    # GUI 슬라이더 매핑
    gui_thresh = params.get('threshold_abs', 190)
    gmm_prob_th = np.clip(gui_thresh / 300.0, 0.5, 0.99)
    mask_radius_dc = int(params.get('mask_radius_dc', 20))

    # 2. Background removal (순수한 데이터에서 수행해야 정확함)
    bg = gaussian_filter(fft_log, sigma=bg_sigma)
    fft_hp = fft_log - bg

    # 3. 좌표계 생성
    h, w = fft_log.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    r_flat = r.ravel()
    fft_hp_flat = fft_hp.ravel()

    # 4. RDF (Radial Distribution Function)
    r_counts = np.bincount(r_flat)
    r_counts[r_counts == 0] = 1
    
    radial_I_hp = np.bincount(r_flat, fft_hp_flat) / r_counts
    
    # [중요] DC Mask 영역은 RDF 계산 후 링 탐색에서 제외하기 위해 값을 죽임
    # (이미지를 자르는 게 아니라 RDF 그래프에서만 앞부분을 무시)
    radial_I_smooth = gaussian_filter1d(radial_I_hp, sigma=rd_sigma)
    if len(radial_I_smooth) > mask_radius_dc:
        radial_I_smooth[:mask_radius_dc] = 0 # DC 영역 무시

    # 링(Ring) 찾기
    peaks_r, _ = find_peaks(radial_I_smooth, prominence=rd_prominence, distance=rd_distance)

    # 5. 분석 영역 마스크(ROI) 설정
    if len(peaks_r) > 0:
        # 가장 강한 링을 기준으로 대역폭 설정
        r0 = peaks_r[np.argmax(radial_I_smooth[peaks_r])]
        ring_mask = (r >= r0 - bw) & (r <= r0 + bw)
        
        # 링이 찾아졌더라도 DC 영역은 확실히 배제
        ring_mask[r < mask_radius_dc] = False
    else:
        # 링을 못 찾았으면 DC만 빼고 전체 영역 사용
        ring_mask = np.ones_like(r, dtype=bool)
        ring_mask[r < mask_radius_dc] = False

    # 6. Radial Z-score (통계적 이탈 정도)
    diff2 = (fft_hp_flat - radial_I_hp[r_flat])**2
    radial_std = np.sqrt(np.bincount(r_flat, diff2) / r_counts)
    
    # 분모 0 방지
    safe_std = radial_std[r]
    safe_std[safe_std == 0] = 1e-6
    
    radial_z = (fft_hp - radial_I_hp[r]) / safe_std
    radial_z[~ring_mask] = 0 # 관심 영역 밖은 0 처리

    # 7. LoG (Laplacian of Gaussian)
    # 링 마스크 내부에서만 LoG 반응 계산
    log_resp = -gaussian_laplace(radial_z, sigma=log_sigma)
    log_resp[log_resp < 0] = 0
    log_resp[~ring_mask] = 0

    # 8. 1차 후보군 탐색 (낮은 문턱값으로 넉넉하게 잡기)
    peaks_candidate = peak_local_max(
        log_resp, 
        min_distance=min_dist, 
        threshold_rel=threshold_rel
    )

    if len(peaks_candidate) < 2:
        return []

    # 9. GMM Clustering (Signal vs Noise)
    try:
        py, px = peaks_candidate[:, 0], peaks_candidate[:, 1]
        
        # Feature: [Z-score, LoG반응, 반지름, 각도(대칭성)]
        theta = np.arctan2(py - cy, px - cx)
        X = np.column_stack([
            radial_z[py, px],
            log_resp[py, px],
            r[py, px],
            np.abs(np.sin(theta))
        ])

        # 정규화 및 GMM
        Xn = StandardScaler().fit_transform(X)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(Xn)
        probs = gmm.predict_proba(Xn)

        # "어느 그룹이 진짜인가?" -> LoG 반응 평균이 높은 쪽이 진짜 Peak임
        mean_log = [log_resp[py[labels == k], px[labels == k]].mean() for k in range(2)]
        signal_cluster = np.argmax(mean_log)
        
        # 확률 기반 필터링 (GUI 슬라이더 값 적용)
        peak_prob = probs[:, signal_cluster]
        final_idx = peak_prob > gmm_prob_th
        
        final_peaks = peaks_candidate[final_idx]
        return final_peaks

    except Exception as e:
        print(f"GMM Error: {e}")
        # GMM 실패 시 그냥 상위 후보군 리턴
        return peaks_candidate


def pair_and_cluster_peaks(peaks, center, symmetry_tol=5.0, tolerance_pct=1.0):
    if len(peaks) == 0: return {}
    crow, ccol = center
    peaks_array = np.array(peaks)
    used_mask = np.zeros(len(peaks), dtype=bool)
    features = []

    for i in range(len(peaks)):
        if used_mask[i]: continue
        p1 = peaks_array[i]
        used_mask[i] = True
        target_r = 2 * crow - p1[0]
        target_c = 2 * ccol - p1[1]
        best_idx = -1
        min_dist = float('inf')
        for j in range(i + 1, len(peaks)):
            if used_mask[j]: continue
            d = np.sqrt((peaks_array[j][0] - target_r)**2 + (peaks_array[j][1] - target_c)**2)
            if d < min_dist:
                min_dist = d
                best_idx = j
        if best_idx != -1 and min_dist <= symmetry_tol:
            p2 = peaks_array[best_idx]
            used_mask[best_idx] = True
            real_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            features.append({'radius': real_dist / 2.0, 'peaks': [tuple(p1), tuple(p2)]})
            
    features.sort(key=lambda x: x['radius'])
    groups = {}
    group_id = 0
    if not features: return {}

    current_group_peaks = []
    current_group_peaks.extend(features[0]['peaks'])
    ref_radius = features[0]['radius']
    
    for i in range(1, len(features)):
        curr_radius = features[i]['radius']
        allowed_diff = max(ref_radius * (tolerance_pct / 100.0), 2.0)
        if abs(curr_radius - ref_radius) <= allowed_diff:
            current_group_peaks.extend(features[i]['peaks'])
        else:
            groups[group_id] = current_group_peaks
            group_id += 1
            current_group_peaks = []
            current_group_peaks.extend(features[i]['peaks'])
            ref_radius = curr_radius 
    if current_group_peaks: groups[group_id] = current_group_peaks
    return groups

def reconstruct_from_groups(fshift, groups, params):
    rows, cols = fshift.shape
    cmap = plt.get_cmap('tab10')
    group_colors = [cmap(i) for i in range(10)]
    
    canvas_ifft = np.zeros((rows, cols, 3), dtype=np.uint8)
    canvas_points = np.zeros((rows, cols, 3), dtype=np.uint8)
    hull_data = [] 

    pixel_size_nm = params.get('pixel_size_nm', 1.0)
    area_scale = pixel_size_nm ** 2

    for gid, g_peaks in groups.items():
        color_rgba = group_colors[gid % 10]
        color_rgb = (np.array(color_rgba[:3]) * 255).astype(np.uint8)

        mask = np.zeros((rows, cols), np.uint8)
        for peak in g_peaks:
            cv2.circle(mask, (peak[1], peak[0]), int(params['mask_radius_peak']), 1, -1)
        
        fshift_filtered = fshift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        img_back = np.abs(img_back)
        
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh_val = np.percentile(img_back_norm, params['pixel_thresh_percent'])
        _, binary_img = cv2.threshold(img_back_norm, thresh_val, 255, cv2.THRESH_BINARY)
        
        canvas_ifft[binary_img > 0] = color_rgb
        
        points = np.column_stack(np.where(binary_img > 0))
        if len(points) == 0: continue
        
        db = DBSCAN(eps=params['dbscan_eps'], min_samples=int(params['dbscan_min_samples'])).fit(points)
        labels = db.labels_
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: continue
            cluster_points = points[labels == label]
            
            for pt in cluster_points:
                canvas_points[pt[0], pt[1]] = color_rgb
            
            if ALPHASHAPE_AVAILABLE:
                try:
                    hull = alphashape.alphashape(cluster_points, params['alpha_value'])
                    if not hull.is_empty:
                        hull_data.append({
                            'geom': hull,
                            'color': color_rgba,
                            'area': hull.area * area_scale,
                            'id': gid,
                            'centroid': hull.centroid
                        })
                except: pass
                
    return canvas_ifft, canvas_points, hull_data, group_colors

def run_processing(raw_img, params):
    # 1. 전처리 (밝기 조절)
    norm_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    brightness = params.get('brightness', 1.0)
    bright_img = norm_img * brightness
    display_img = np.clip(bright_img, 0, 255).astype(np.uint8)

    rows, cols = bright_img.shape
    crow, ccol = rows // 2, cols // 2
    
    # 2. FFT (CLAHE나 DC Masking을 하지 않은 순수 Magnitude 사용)
    f = np.fft.fft2(bright_img)
    fshift = np.fft.fftshift(f) 
    magnitude = np.log1p(np.abs(fshift))
    
    # [수정됨] GMM 알고리즘에 '순수한 Magnitude' 전달 (내부에서 통계처리 및 Masking 수행)
    peaks = detect_peaks_advanced(magnitude, params)
    
    # 3. Grouping
    groups = pair_and_cluster_peaks(peaks, (crow, ccol), symmetry_tol=10.0, tolerance_pct=params['tolerance_pct'])
    
    # 4. D-Spacing
    pixel_size_nm = params.get('pixel_size_nm', 1.0)
    d_spacings = {}
    if pixel_size_nm > 0:
        recip_scale = 1.0 / (cols * pixel_size_nm)
        for gid, g_peaks in groups.items():
            radii = []
            for p in g_peaks:
                r_px = np.sqrt((p[0] - crow)**2 + (p[1] - ccol)**2)
                radii.append(r_px)
            if radii:
                avg_r_px = np.mean(radii)
                g_vector_mag = avg_r_px * recip_scale
                if g_vector_mag > 0:
                    d_spacings[gid] = 1.0 / g_vector_mag
                else:
                    d_spacings[gid] = 0.0

    # 5. Reconstruction
    canvas_ifft, canvas_points, hull_data, group_colors = reconstruct_from_groups(fshift, groups, params)

    return {
        'fshift': fshift,
        'magnitude': magnitude,
        'groups': groups,
        'group_colors': group_colors,
        'canvas_ifft': canvas_ifft,
        'canvas_points': canvas_points,
        'hull_data': hull_data,
        'mask_radius_peak': int(params['mask_radius_peak']),
        'display_img': display_img,
        'd_spacings': d_spacings
    }