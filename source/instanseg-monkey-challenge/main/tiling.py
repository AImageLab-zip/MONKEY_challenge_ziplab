
import numpy as np

def is_overlapping(new_pos, existing_positions, tile_size):
    new_x, new_y = new_pos
    for (existing_x, existing_y) in existing_positions:
        if (new_x < existing_x + tile_size and new_x + tile_size > existing_x and
            new_y < existing_y + tile_size and new_y + tile_size > existing_y):
            return True
    return False


def find_non_empty_positions(mask, tile_size, downsample_factor_mask, threshold = 0.5):
    """
    Precompute all valid positions within the mask where tiles can be placed.
    """
    valid_positions = []
    scaled_tile_size = int(tile_size * (1 / downsample_factor_mask))
    mask_height, mask_width = mask.shape

    for y in range(0, mask_height - scaled_tile_size, scaled_tile_size):
        for x in range(0, mask_width - scaled_tile_size, scaled_tile_size):
            # Check if the region in the mask contains any valid area
            if mask[y:y + scaled_tile_size, x:x + scaled_tile_size].mean() > threshold:
                valid_positions.append((x, y))
    
    return valid_positions


def generate_random_tiles_from_valid_positions(valid_positions, num_images, tile_size, downsample_factor_mask):
    """
    Select random non-overlapping tiles from precomputed valid positions.
    """
    scaled_tile_size = int(tile_size * (1 / downsample_factor_mask))
    selected_positions = []
    np.random.shuffle(valid_positions)  # Shuffle positions to pick randomly

    for pos in valid_positions:
        if len(selected_positions) >= num_images:
            break

        new_x, new_y = pos
        # Scale the positions back to the original coordinate system
        scaled_pos = (int(new_x * downsample_factor_mask), int(new_y * downsample_factor_mask))
        if not is_overlapping(scaled_pos, selected_positions, tile_size):
            selected_positions.append(scaled_pos)

    return selected_positions


def threshold_thumbnail(slide, level=None):
    from skimage.color import rgb2gray
    from skimage import filters
    import numpy as np

    if level is None:
        level = slide.level_count - 1

    img_thumbnail = slide.read_region((0, 0), level, size=(10000, 10000), as_array=True, padding=False)
    downsample_factor_thumbnail = slide.level_downsamples[level]

    gray_image = rgb2gray(np.array(img_thumbnail))
    threshold_value = filters.threshold_otsu(gray_image)
    binary_image = ~(gray_image > threshold_value)  # Apply the threshold to create a binary image

    return binary_image, downsample_factor_thumbnail


def get_tiles(slide, positions, tile_size=512, level=None):
    if level is None:
        level = 0

    tiles = []
    for posx, posy in positions:
        img = slide.read_region((posx, posy), level, (tile_size, tile_size), padding=False, as_array=True)
        tiles.append(img)
    return tiles


def get_random_non_empty_tiles(slide, slide2= None, num_images=100, tile_size=512):
    x_max, y_max = slide.dimensions
    binary_image, downsample_factor_thumbnail = threshold_thumbnail(slide)
    valid_positions = find_non_empty_positions(binary_image, tile_size=tile_size, downsample_factor_mask=downsample_factor_thumbnail)
    positions = generate_random_tiles_from_valid_positions(valid_positions, num_images=num_images, tile_size=tile_size, downsample_factor_mask=downsample_factor_thumbnail)

    tiles = get_tiles(slide, positions, tile_size=tile_size, level=0)

    if slide2 is not None:
        tiles2 = get_tiles(slide2, positions, tile_size=tile_size, level=0)
        return tiles, tiles2
    else:
        return tiles