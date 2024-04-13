import cv2
import cv2
import numpy as np



def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Coordenada: ({x}, {y})')


def region_grow(image, seed_coords, threshold_min, threshold_max):
    # Create an empty mask to store the region
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Create a set to store the coordinates of pixels that have been processed
    processed_pixels = set()
    
    # Get the seed pixel value
    seed_value = image[seed_coords[1], seed_coords[0]]
    
    # Create a queue to store the coordinates of pixels to be processed
    queue = []
    queue.append(seed_coords)
    
    # Process the queue until it's empty
    while queue:
        # Get the next pixel coordinates from the queue
        x, y = queue.pop(0)
        
        # Check if the pixel is within the image boundaries and has not been processed before
        if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0] and (x, y) not in processed_pixels:
            # Mark the pixel as processed
            processed_pixels.add((x, y))
            # Check if the pixel value is within the threshold
            if threshold_min <= image[y, x][0] <= threshold_max:
                
                # Set the pixel value in the mask
                mask[y, x] = 255
                
                # Add the neighboring pixels to the queue
                queue.append((x - 1, y))
                queue.append((x + 1, y))
                queue.append((x, y - 1))
                queue.append((x, y + 1))
        '''cv2.imshow("Mask", mask)
        cv2.waitKey(1)'''

    return mask

# Path to the image file
image_path = "MaterialP2/hueso.tif"

# Read the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is not None:
    # Display the image
    cv2.imshow("Image", image)
    # Establecer la función de devolución de llamada para el evento de clic del mouse
    cv2.setMouseCallback('Image', click_event)
    # Wait for a key press
    cv2.waitKey(0)
    
    '''
    seed_x = int(input("Enter the x coordinate of the seed pixel: "))
    seed_y = int(input("Enter the y coordinate of the seed pixel: "))'''

    seed_x = 153
    seed_y = 281
    print(seed_x, seed_y)

    # Get the threshold from the user
    '''
    threshold_max = int(input("Enter the threshold value max: "))
    threshold_min = int(input("Enter the threshold value min: "))
    '''



    threshold_max = 180 # 220 te detecta mas hueso
    threshold_min = 110
    
    # Perform region growing
    mask = region_grow(image, (seed_x, seed_y), threshold_min, threshold_max)
    
    # Display the mask
    cv2.imshow("Mask", mask)
    
    # Wait for a key press
    cv2.waitKey(0)
    
    # Close all windows
    cv2.destroyAllWindows()
else:
    print("Failed to load the image.")

