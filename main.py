import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from mesh_segment import mesh
from warp_mesh import warp_mesh
from line_preservation import El
from shape_preservation import Es
from boundary_preservation import Eb
from line_preservation import get_pk, get_u
from lsd_detect_lines import detect_lines
from optimization import fix_theta_solve_for_v, fix_v_solve_for_theta


def illustrate_lines(img_array, detected_lines, angle = 0,output_path="output_lines.jpg"):
    """
    Marks the detected lines on the image and saves the result.
    """
    img_with_lines = img_array.copy()

    # Convert grayscale images to RGB
    if len(img_with_lines.shape) == 2:
        img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_GRAY2RGB)

    for line in detected_lines:
        x1, y1, x2, y2, theta, bin_num = line
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        
        # if bin_num in [1, 45, 46, 90]:
            # color = (255, 0, 0)  # Red for highlighted bins
        # else:
            # color = (0, 255, 0)  # Green for other lines
        if bin_num in [1, 90]:
            color = (255, 0, 0)
        elif bin_num in [45, 46]:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        
        # Draw lines
        cv2.line(img_with_lines, start_point, end_point, color, 2)
        
        # Annotate with angle and bin number
        # text = f"θ:{int(theta)}° Bin:{bin_num}"
        # cv2.putText(img_with_lines, text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Save the result
    cv2.imwrite(output_path, img_with_lines)
    print(f"Lines illustrated and saved to {output_path}")

    return img_with_lines


def visualize_results(original_img, rotated_img, initial_mesh, warped_mesh, lines_marked_img, rotated_lines_marked_img):
    """
    Visualizes the images and meshes using matplotlib.
    """
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # Lines marked image
    plt.subplot(2, 3, 2)
    plt.imshow(lines_marked_img)
    plt.title("Detected Lines Marked")
    plt.axis("off")

    # Rotated image
    plt.subplot(2, 3, 3)
    plt.imshow(rotated_img)
    plt.title("Content Aware Rotated Image")
    plt.axis("off")

   # Lines marked on rotated image
    plt.subplot(2, 3, 4)
    plt.imshow(rotated_lines_marked_img)
    plt.title("Detected Lines Marked (Rotated)")
    plt.axis("off")

    # Initial mesh
    plt.subplot(2, 3, 5)
    plt.imshow(initial_mesh)
    plt.title("Initial Mesh")
    plt.axis("off")

    # Warped mesh
    plt.subplot(2, 3, 6)
    plt.imshow(warped_mesh)
    plt.title("Warped Mesh")
    plt.axis("off")

    # plt.tight_layout()
    plt.show()


def image_rotation(img_path, angle, M=90, lambda_r=1e2, lambda_b=1e8, lambda_l=1e2, lambda_s=1):
    """
    Content-aware image rotation pipeline.
    """
    # Load the image
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)

    # Initialize the mesh
    x_len, y_len, x, y, grid_x, grid_y, nv, initial_mesh = mesh(img_array.copy())

    # Initialize rotation parameters
    thetas = np.ones((M,)) * angle
    w_thetas = np.zeros((M,))
    index = [0, M - 1, M // 2 - 1, M // 2]
    # modify
    # index2 = [0, M - 1, M // 2 - 1, M // 2]
    w_thetas[index] = 1e3

    # Detect lines
    lines = detect_lines(img_array, angle, x_len, y_len, x, y, M=M)

    # Mark detected lines on the image
    lines_marked_img = illustrate_lines(img_array, lines, output_path="marked_lines.jpg")

    # Boundary preservation
    dEs = Es(grid_x, grid_y)
    G, b = Eb(grid_x, grid_y, x, y)

    # Line preservation
    Pk = get_pk(lines, x_len, y_len, grid_x, grid_y)
    U = get_u(lines)

    # Optimization loop
    for epoch in range(10):
        print(f'\nEpoch: {epoch}')
        dEl, total_Uk = El(lines, U, Pk, nv, thetas)

        print(' ' * 6 + 'Optimization: Fix theta, solve for V')
        V = fix_theta_solve_for_v(dEl, lambda_l, G, b, lambda_b, dEs, lambda_s)

        print(' ' * 6 + 'Optimization: Fix V, solve for theta')
        thetas = fix_v_solve_for_theta(lines, angle, w_thetas, thetas, lambda_r, lambda_l, Pk, V, total_Uk)

    # Warp the mesh and image
    mesh_result, img_result = warp_mesh(initial_mesh, img_array, V, grid_x, grid_y)
    rgb_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/image00.jpg", rgb_result)

    rotated_lines = detect_lines(img_result, 0, x_len, y_len, x, y, M=M)

    rotated_lines_marked_img = illustrate_lines(img_result, rotated_lines, angle, output_path="rotated_marked_lines.jpg")
    # Visualize results
    visualize_results(img_array, img_result, initial_mesh, mesh_result, lines_marked_img, rotated_lines_marked_img)


if __name__ == '__main__':
    img_path = './sharpened_image/image2.png'
    angle = -6.1
    image_rotation(img_path, angle)
