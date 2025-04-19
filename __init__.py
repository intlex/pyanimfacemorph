import cv2
import numpy as np
import face_alignment
import imageio
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
import torch

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    face_detector='sfd',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    flip_input=False
)

def get_landmarks_face(image):
    preds = fa.get_landmarks(image)
    if preds is None:
        raise Exception("No face found!")
    return preds[0].astype(int).tolist()

def rect_contains(rect, point):
    x, y, w, h = rect
    return (point[0] >= x and point[0] <= x+w and point[1] >= y and point[1] <= y+h)

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    pts = np.array(points)
    for t in triangleList:
        pts_tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if rect_contains(rect, pts_tri[0]) and rect_contains(rect, pts_tri[1]) and rect_contains(rect, pts_tri[2]):
            indices = []
            for pt in pts_tri:
                index = np.where((pts == pt).all(axis=1))[0]
                if len(index) == 0:
                    continue
                indices.append(index[0])
            if len(indices) == 3:
                delaunayTri.append(tuple(indices))
    return delaunayTri

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_rect  = [(t[i][0] - r[0],  t[i][1] - r[1]) for i in range(3)]
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    size_rect = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size_rect)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size_rect)
    img_rect = (1 - alpha) * warp_img1 + alpha * warp_img2
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

def morph_faces_with_cache(face1_img, face2_img, alpha, landmarks1, landmarks2, dt):
    h, w = face1_img.shape[:2]

    interm_points = [((1 - alpha) * p1[0] + alpha * p2[0],
                      (1 - alpha) * p1[1] + alpha * p2[1])
                     for p1, p2 in zip(landmarks1, landmarks2)]
    
    boundary_pts = [(0, 0), (w//2, 0), (w-1, 0), (w-1, h//2),
                    (w-1, h-1), (w//2, h-1), (0, h-1), (0, h//2)]
    landmarks1_ext = landmarks1 + boundary_pts
    landmarks2_ext = landmarks2 + boundary_pts

    morphed_img = face1_img.copy().astype(np.float32)
    for tri in dt:
        i1, i2, i3 = tri
        t1 = [landmarks1_ext[i1], landmarks1_ext[i2], landmarks1_ext[i3]]
        t2 = [landmarks2_ext[i1], landmarks2_ext[i2], landmarks2_ext[i3]]
        t  = [((1 - alpha) * landmarks1_ext[i1][0] + alpha * landmarks2_ext[i1][0],
                (1 - alpha) * landmarks1_ext[i1][1] + alpha * landmarks2_ext[i1][1]),
              ((1 - alpha) * landmarks1_ext[i2][0] + alpha * landmarks2_ext[i2][0],
                (1 - alpha) * landmarks1_ext[i2][1] + alpha * landmarks2_ext[i2][1]),
              ((1 - alpha) * landmarks1_ext[i3][0] + alpha * landmarks2_ext[i3][0],
                (1 - alpha) * landmarks1_ext[i3][1] + alpha * landmarks2_ext[i3][1])]
        morph_triangle(face1_img, face2_img, morphed_img, t1, t2, t, alpha)
    
    # Calculate the mask only from the landmarks of the first image 
    # to limit the morphing area to the position of the face in the first image.
    face_hull = cv2.convexHull(np.array(landmarks1, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, face_hull, 255)
    # Expand and smooth the mask to create a smooth transition zone.
    dilate_multiplier = int(alpha * 300)
    dilate_multiplier = max(1, dilate_multiplier)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_multiplier, dilate_multiplier))
    mask_dilated = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask_dilated_3ch = cv2.merge([mask_dilated]*3).astype(np.float32) / 255.0
    blend_multiplier = 250
    ksize = int(alpha * blend_multiplier)
    ksize = ksize if ksize % 2 == 1 else ksize+1
    ksize = max(3, ksize)
    mask_smooth = cv2.GaussianBlur(mask_dilated_3ch, (ksize, ksize), 0)
    # Calculate the background as a linear interpolation between the first image and the morphed result, 
    # which ensures a smooth transition from face1 (stored in position) to morphed.
    face1_float = face1_img.astype(np.float32)
    background = (1 - alpha) * face1_float + alpha * morphed_img
    final_img = background * (1 - mask_smooth) + morphed_img * mask_smooth
    final_img_rgb = cv2.cvtColor(np.uint8(final_img), cv2.COLOR_BGR2RGB)
    return final_img_rgb, interm_points

def composite_landmark_overlay(image, face1_landmarks, face2_landmarks, interm_landmarks):
    overlay = image.copy()
    for (x, y) in face1_landmarks:
        cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 0), -1)
    for (x, y) in face2_landmarks:
        cv2.circle(overlay, (int(x), int(y)), 2, (0, 0, 255), -1)
    for (x, y) in interm_landmarks:
        cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)
    return overlay

def gradio_morph_animation(face1_img, face2_img):
    # Convert to BGR
    face1_img_bgr = cv2.cvtColor(face1_img, cv2.COLOR_RGB2BGR)
    face2_img_bgr = cv2.cvtColor(face2_img, cv2.COLOR_RGB2BGR)
    h, w = face1_img_bgr.shape[:2]
    face2_img_bgr = cv2.resize(face2_img_bgr, (w, h))
    
    landmarks1 = get_landmarks_face(face1_img_bgr)
    landmarks2 = get_landmarks_face(face2_img_bgr)
    
    boundary_pts = [(0, 0), (w//2, 0), (w-1, 0), (w-1, h//2),
                    (w-1, h-1), (w//2, h-1), (0, h-1), (0, h//2)]
    landmarks1_ext = landmarks1 + boundary_pts
    landmarks2_ext = landmarks2 + boundary_pts
    rect = (0, 0, w, h)
    avg_points = [((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                  for p1, p2 in zip(landmarks1_ext, landmarks2_ext)]
    dt = calculate_delaunay_triangles(rect, avg_points)
    
    frames = []
    n_frames = 20
    alphas = np.linspace(0, 1, n_frames)
    for a in alphas:
        final_img_rgb, _ = morph_faces_with_cache(face1_img_bgr, face2_img_bgr, a, landmarks1, landmarks2, dt)
        frames.append(final_img_rgb)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    imageio.mimsave(temp_file.name, frames, format="GIF", duration=0.1)
    
    # Show checkpoints
    mid_img_rgb, interm_points = morph_faces_with_cache(face1_img_bgr, face2_img_bgr, 0.5, landmarks1, landmarks2, dt)
    composite_overlay_img = composite_landmark_overlay(mid_img_rgb, landmarks1, landmarks2, interm_points)
    
    message = "The animation was successful!"
    return temp_file.name, composite_overlay_img, message


demo = gr.Interface(
    fn=gradio_morph_animation,
    inputs=[
        gr.Image(label="First face", type="numpy"),
        gr.Image(label="Last face", type="numpy")
    ],
    outputs=[
        gr.Image(label="Animation (GIF)"),
        gr.Image(label="Checkpoints (α=0.5)"),
        gr.Textbox(label="Message", lines=2)
    ],
    title="Animated facial morphing",
    description="Upload images – animation is created by changing α from 0 to 1, and checkpoints are shown for a frame with α=0.5."
)

demo.launch(debug=True)
