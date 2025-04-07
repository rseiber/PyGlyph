import os
import pygame
import colorsys
import numpy as np
from math import sin, cos, sqrt
from numba import jit, njit, prange

def load_obj(filename):
    """Loads a Wavefront OBJ file and returns vertices and faces."""
    vertices = []
    faces = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face = []
                    for p in parts:
                        vert_index_str = p.split("/")[0]
                        face.append(int(vert_index_str) - 1)
                    faces.append(face)
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return [], []
        
    return np.array(vertices, dtype=np.float32), faces

@njit
def compute_face_normal(v0, v1, v2):
    """Calculate normal vector for a triangle face."""
    x1, y1, z1 = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    x2, y2, z2 = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx = y1*z2 - z1*y2
    ny = z1*x2 - x1*z2
    nz = x1*y2 - y1*x2
    return np.array([nx, ny, nz], dtype=np.float32)

@njit
def normalize(vec):
    """Normalize a vector to unit length."""
    length = np.sqrt(np.sum(vec * vec))
    if length < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return vec / length

def compute_vertex_normals(vertices, faces):
    """Compute smooth vertex normals by averaging adjacent face normals."""
    v_normals = np.zeros((len(vertices), 3), dtype=np.float32)
    
    # First compute face normals
    face_normals = []
    for face in faces:
        if len(face) < 3:
            face_normals.append(np.zeros(3, dtype=np.float32))
            continue
            
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        fn = compute_face_normal(v0, v1, v2)
        fn = normalize(fn)
        face_normals.append(fn)
    
    # Build vertex-face association
    vertex_faces = [[] for _ in range(len(vertices))]
    for i, face in enumerate(faces):
        for v_idx in face:
            vertex_faces[v_idx].append(i)
    
    # Average face normals for each vertex
    for i in range(len(vertices)):
        if not vertex_faces[i]:
            continue
            
        normals = [face_normals[f_idx] for f_idx in vertex_faces[i]]
        avg_normal = np.zeros(3, dtype=np.float32)
        for n in normals:
            avg_normal += n
            
        v_normals[i] = normalize(avg_normal)
    
    return v_normals

@njit(parallel=True)
def render_vertices(vertices, v_normals, light_dir, chars, zbuffer, output, 
                    screen_cols, screen_rows, K1, dist_to_camera, cosA, sinA, cosB, sinB):
    """Render vertices with shading based on surface normals."""
    char_len = len(chars) - 1
    z_epsilon = 0.00001
    
    for i in prange(len(vertices)):
        vx, vy, vz = vertices[i]
        
        # Rotate vertex around X, then Y
        y2 = vy*cosA - vz*sinA
        z2 = vy*sinA + vz*cosA
        x3 = vx*cosB + z2*sinB
        z3 = -vx*sinB + z2*cosB

        # Rotate normal similarly
        nx, ny, nz = v_normals[i]
        ny2 = ny*cosA - nz*sinA
        nz2 = ny*sinA + nz*cosA
        nx3 = nx*cosB + nz2*sinB
        nz3 = -nx*sinB + nz2*cosB

        # Calculate lighting with ambient component
        lum = nx3*light_dir[0] + ny2*light_dir[1] + nz3*light_dir[2]
        lum = lum * 0.8 + 0.2  # Add 20% ambient light
        lum = max(0.05, min(0.95, lum))
        
        # Get character index
        idx = max(1, min(char_len-1, int(lum * char_len)))

        # Calculate perspective projection
        z_trans = max(z_epsilon, z3 + dist_to_camera)
        ooz = 1.0 / z_trans

        # Project to screen coordinates
        xp = int(screen_cols*0.42 + (K1*x3*ooz))
        yp = int(screen_rows*0.5 - (K1*y2*ooz))

        if 0 <= xp < screen_cols and 0 <= yp < screen_rows:
            pos = xp + screen_cols*yp
            
            # Add tiny bias to prevent z-fighting
            adjusted_ooz = ooz + (i * 0.000000001)
            
            if adjusted_ooz > zbuffer[pos]:
                zbuffer[pos] = adjusted_ooz
                output[pos] = idx

def hsv2rgb(h, s, v):
    """Convert HSV color to RGB."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def main():
    # Setup display
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    FLAGS = pygame.HWSURFACE | pygame.DOUBLEBUF

    WIDTH, HEIGHT = 1200, 900
    FPS = 60

    pixel_width = 10
    pixel_height = 10

    # ASCII resolution
    screen_cols = WIDTH // pixel_width
    screen_rows = HEIGHT // pixel_height
    screen_size = screen_cols * screen_rows

    # ASCII gradient characters
    chars = ".,-~:;=!*#$@"
    char_indices = list(range(len(chars)))

    # Initial states
    A, B = 0.0, 0.0  # Rotation angles
    hue = 0.0

    # Light direction
    light_dir = normalize(np.array([0.5, 0.5, -1.0], dtype=np.float32))

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), FLAGS)
    pygame.display.set_caption("3D ASCII Renderer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 22, bold=False)

    # Pre-render characters for performance
    char_surfaces = {}
    for idx, char in enumerate(chars):
        char_surfaces[idx] = font.render(char, True, (255, 255, 255))

    # Load and prepare the model
    try:
        model_file = "model.obj"
        vertices, faces = load_obj(model_file)
        if len(vertices) == 0:
            raise RuntimeError(f"No vertices found in OBJ file: {model_file}")
            
        v_normals = compute_vertex_normals(vertices, faces)

        # Center the model
        centroid = np.mean(vertices, axis=0)
        vertices -= centroid
        
        # Apply x-offset to center horizontally
        x_correction = np.array([-0.4, 0.0, 0.0], dtype=np.float32)
        vertices += x_correction
        
        # Calculate bounding radius and camera distance
        R = np.max(np.sqrt(np.sum(vertices * vertices, axis=1)))
        dist_to_camera = max(5.0 * R, 3.0)
        
        # Calculate projection scale
        if R > 1e-6:
            max_k1_for_width = 0.4 * screen_cols * dist_to_camera / R
            max_k1_for_height = 0.4 * screen_rows * dist_to_camera / R
            K1 = min(max_k1_for_width, max_k1_for_height)
        else:
            K1 = 100.0
        
        print(f"Model '{model_file}' loaded: {len(vertices)} vertices, {len(faces)} faces")
        print(f"Radius: {R:.2f}, Camera Distance: {dist_to_camera:.2f}, Scale: {K1:.2f}")
    
    except Exception as e:
        print(f"Error setting up model: {e}")
        pygame.quit()
        return

    # Create buffers
    zbuffer = np.full(screen_size, -999999.0, dtype=np.float32)
    output = np.zeros(screen_size, dtype=np.int32)
    row_starts = [row * screen_cols for row in range(screen_rows)]
    row_surfaces = {}
    
    # Main loop
    running = True
    paused = False
    frame_count = 0
    last_time = pygame.time.get_ticks()
    
    while running:
        # Handle timing
        clock.tick_busy_loop(FPS)
        current_time = pygame.time.get_ticks()
        frame_count += 1
        
        # Update FPS counter every second
        if current_time - last_time >= 1000:
            fps_val = frame_count / ((current_time - last_time) / 1000.0)
            pygame.display.set_caption(f"3D ASCII (FPS: {fps_val:.1f})")
            frame_count = 0
            last_time = current_time

        # Clear screen and reset buffers
        screen.fill((0, 0, 0))
        zbuffer.fill(-999999.0)
        output.fill(0)  # 0 = background

        # Precompute rotation values
        cosA, sinA = cos(A), sin(A)
        cosB, sinB = cos(B), sin(B)

        # Render the model
        render_vertices(vertices, v_normals, light_dir, char_indices, zbuffer, output, 
                        screen_cols, screen_rows, K1, dist_to_camera, cosA, sinA, cosB, sinB)

        # Display characters
        color = hsv2rgb(hue, 1.0, 1.0)
        
        for row in range(screen_rows):
            start = row_starts[row]
            end = start + screen_cols
            row_data = output[start:end]
            
            # Skip empty rows
            if np.all(row_data == 0):
                continue
                
            # Create row string with spaces for background
            row_str = ""
            for idx in row_data:
                if idx > 0:
                    row_str += chars[idx]
                else:
                    row_str += " "
            
            # Cache and render row
            row_key = (row_str, color)
            if row_key not in row_surfaces:
                row_surfaces[row_key] = font.render(row_str, True, color)
                
                # Limit cache size
                if len(row_surfaces) > 1000:
                    row_surfaces = dict(list(row_surfaces.items())[-500:])
            
            screen.blit(row_surfaces[row_key], (0, row*pixel_height))
        
        # Update rotation and color
        if not paused:
            A += 0.02
            B += 0.03
            hue += 0.005
            if hue > 1.0:
                hue = 0.0

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused

    pygame.quit()

if __name__ == "__main__":
    main()