import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

def bilinear_sample(image, map_x, map_y):
    """
    Bilinear interpolation sampling.
    image: H x W x C
    map_x, map_y: H x W, float32
    """
    h, w = image.shape[:2]

    if h == 1 or w == 1:
        map_x = np.clip(np.rint(map_x).astype(np.int32), 0, w - 1)
        map_y = np.clip(np.rint(map_y).astype(np.int32), 0, h - 1)
        return image[map_y, map_x]

    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = map_x - x0
    dy = map_y - y0

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (1.0 - dx) * (1.0 - dy)
    wb = (1.0 - dx) * dy
    wc = dx * (1.0 - dy)
    wd = dx * dy

    if image.ndim == 3:
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return np.clip(out, 0, 255).astype(np.uint8)

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1e-4, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping
    if image is None:
        return None

    warped_image = np.array(image).copy()
    h, w = warped_image.shape[:2]

    if len(source_pts) == 0 or len(target_pts) == 0:
        return warped_image

    n = min(len(source_pts), len(target_pts))
    source_pts = np.asarray(source_pts[:n], dtype=np.float32)
    target_pts = np.asarray(target_pts[:n], dtype=np.float32)

    if n == 1:
        # 只有一个点时，退化成整体平移
        shift = source_pts[0] - target_pts[0]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = grid_x.astype(np.float32) + shift[0]
        map_y = grid_y.astype(np.float32) + shift[1]
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        return bilinear_sample(warped_image, map_x, map_y)

    # 位移定义在 target 点上：目标图中的点，应该回到 source 点采样
    disp = source_pts - target_pts  # shape: (n, 2)

    def phi(r2):
        # 常用高斯 RBF
        return np.exp(-alpha * r2)

    # 构建控制点核矩阵 K
    diff = target_pts[:, None, :] - target_pts[None, :, :]   # (n, n, 2)
    r2 = np.sum(diff ** 2, axis=2)                           # (n, n)
    K = phi(r2) + eps * np.eye(n, dtype=np.float32)

    # 求解 x/y 两个方向的权重
    wx = np.linalg.solve(K, disp[:, 0])
    wy = np.linalg.solve(K, disp[:, 1])

    # 生成目标图坐标网格
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    grid = np.stack([grid_x, grid_y], axis=-1)  # (h, w, 2)

    # 计算每个像素到 target 控制点的 RBF 权重
    grid_diff = grid[:, :, None, :] - target_pts[None, None, :, :]   # (h, w, n, 2)
    grid_r2 = np.sum(grid_diff ** 2, axis=3)                         # (h, w, n)
    G = phi(grid_r2)                                                 # (h, w, n)

    # 插值得到每个像素的反向位移
    disp_x = np.tensordot(G, wx, axes=([2], [0]))  # (h, w)
    disp_y = np.tensordot(G, wy, axes=([2], [0]))  # (h, w)

    # inverse mapping: target -> source
    map_x = grid_x + disp_x
    map_y = grid_y + disp_y

    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    warped_image = bilinear_sample(warped_image, map_x, map_y)
    return warped_image
    

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
