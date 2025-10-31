import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time
import os
import sys
from PIL import Image

def load_image(image_path):
    """Loads and preprocesses an image file into RGB format.

    Args:
        image_path (str): Path to the image file to load.

    Returns:
        PIL.Image: The loaded image in RGB format.

    Raises:
        ValueError: If image_path is empty.
        FileNotFoundError: If image file doesn't exist.
        IOError: If there are issues loading/processing the image.
    """
    if not image_path:
        raise ValueError("Image path cannot be empty.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    try:
        # Load the image using PIL and convert to 3-channel RGB for consistency
        image_pil = Image.open(image_path).convert("RGB")

        print(f"Loaded image from: {image_path}")
        return image_pil

    except Exception as e:
        # Catch issues like bad file format or I/O errors
        raise IOError(f"Error loading or processing image: {e}")

def resize_image_aspect_ratio(image_pil, target_width_pixels):
    """Resizes an image to a target width while maintaining aspect ratio.

    Args:
        image_pil (PIL.Image): Input PIL image object.
        target_width_pixels (int): Desired width in pixels.

    Returns:
        numpy.ndarray: Resized image as normalized float array (0-1).

    Raises:
        ValueError: If target_width_pixels is not positive.
    """
    original_width, original_height = image_pil.size

    # Ensure target_width_pixels is an integer for PIL's resize function
    target_width_pixels = int(target_width_pixels)

    if target_width_pixels <= 0:
        raise ValueError(f"Target width must be a positive integer. Got: {target_width_pixels}")

    if original_width == target_width_pixels:
        print("Image width already matches target width. No resizing performed.")
        # Convert to float array (0-1) for consistency with the rest of the pipeline
        return np.array(image_pil).astype(float) / 255.0

    # Calculate new height to maintain the original aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(target_width_pixels * aspect_ratio)

    print(f"Resizing image: Original size ({original_width}px x {original_height}px) -> Target size ({target_width_pixels}px x {new_height}px)")

    # Perform resizing using the high-quality LANCZOS filter
    resized_image_pil = image_pil.resize((target_width_pixels, new_height), Image.Resampling.LANCZOS)

    # Convert the resized PIL image back to a normalized float array (0-1)
    resized_array = np.array(resized_image_pil).astype(float) / 255.0
    return resized_array

def reduce_colors(image_array, n_colors):
    """Reduces image colors using K-Means clustering and sorts by brightness.

    Clusters colors using K-Means and sorts the resulting palette by perceived
    brightness (Luma) from lightest to darkest. Also calculates area coverage
    for each color.

    Args:
        image_array (numpy.ndarray): Input image as float array (0-1).
        n_colors (int): Number of colors to reduce to.

    Returns:
        tuple: (reduced_image, labels, rows, cols) where:
            - reduced_image: Color-reduced image array
            - labels: Cluster labels for each pixel
            - rows: Number of image rows
            - cols: Number of image columns
    """
    t0 = time()

    # Reshape the 3D image array into a 2D array of pixels (N_pixels x 3 color channels)
    rows, cols, channels = image_array.shape
    pixel_data = image_array.reshape(rows * cols, channels)

    print(f"\nStarting K-Means clustering with {n_colors} colors...")
    print(f"Input data shape for K-Means: {pixel_data.shape}")

    # Apply K-Means clustering to find the optimal 'n_colors'
    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42, verbose=0).fit(pixel_data)

    # Get the initial cluster centers (the generated color palette)
    initial_palette = kmeans.cluster_centers_

    # Sort colors from light to dark based on Luma
    # Standard formula for Luma to determine perceived brightness of RGB colors: (Y): Y = 0.299R + 0.587G + 0.114B
    lumas = initial_palette @ np.array([0.299, 0.587, 0.114])

    # Get indices that sort the lumas in DESCENDING order (light to dark)
    sort_indices = np.argsort(lumas)[::-1]

    # Apply the sorting to the palette
    sorted_palette = initial_palette[sort_indices]

    # Create a mapping from old label index (from K-Means) to new sorted label index
    # sort_indices is ORDER-CENTRIC: its index is the NEW rank (0=lightest) and its value is the OLD label (e.g., [2, 0, 1] means OLD label 2 goes into NEW position 0); old_to_new_index_map is LABEL-CENTRIC: its index is the OLD label, and its value is the NEW rank (e.g., [1, 2, 0] means OLD label 2 gets NEW rank 0)
    old_to_new_index_map = np.zeros(n_colors, dtype=int)
    for new_index, old_index in enumerate(sort_indices):
        old_to_new_index_map[old_index] = new_index

    # Get the original cluster label for each pixel
    labels = kmeans.predict(pixel_data)

    # Re-index labels according to the sorted palette
    new_labels = old_to_new_index_map[labels]

    # Reconstruct the image using the new palette
    reduced_image_array = sorted_palette[new_labels]

    # Reshape the 2D pixel array back into the 3D image structure
    reduced_image = reduced_image_array.reshape(rows, cols, channels)

    dt = time() - t0
    print(f"Color reduction completed in {dt:.2f} seconds.")
    print(f"Final palette size: {sorted_palette.shape[0]} colors (Sorted Light to Dark).")

    # Count the number of pixels assigned to each cluster
    label_counts = np.bincount(new_labels)

    print("\n--- Final Color Palette and Pixel Area Log ---")

    # Convert normalized float palette [0, 1] to integer RGB [0, 255] for logging
    int_palette = (sorted_palette * 255).astype(int)

    for i in range(n_colors):
        color_rgb = tuple(int_palette[i])
        count = label_counts[i]

        # Custom area calculation: (Pixel Count * 4) / 100
        meters_value = (count * 4) / 100

        # Log the color, pixel count, and custom area in meters.
        # Color index (i+1) corresponds to the sorted light-to-dark order.
        print(f"Color {i+1:2}: RGB={color_rgb}, Pixels={count:9}, Area={meters_value:7.4f} meters")

    return reduced_image, new_labels, rows, cols

def format_consecutive_numbers(numbers):
    """Converts a list of integers into a condensed range representation.

    Args:
        numbers (list): List of integers to format.

    Returns:
        str: Formatted string with ranges (e.g., "1, 2, 5 ... 8, 10").
    """
    if not numbers:
        return ""

    segments = []
    start = numbers[0]

    # Iterate through the numbers, checking for gaps to define segments
    for i in range(1, len(numbers)):
        # If a gap is found, the previous sequence has ended
        if numbers[i] != numbers[i-1] + 1:
            end = numbers[i-1]
            # Check if the sequence length is 3 or more
            if end - start >= 2:
                segments.append(f"{start} ... {end}")
            else: # Length 1 or 2
                for j in range(start, end + 1):
                    segments.append(str(j))

            # Start a new sequence
            start = numbers[i]

    # Handle the final segment
    end = numbers[-1]
    if end - start >= 2: # Length 3 or more
        segments.append(f"{start} ... {end}")
    else: # Length 1 or 2
        for j in range(start, end + 1):
            segments.append(str(j))

    # Join the segments
    return ', '.join(segments)

# Generates a log of pixel color indices, reading the image from bottom-right (Row Index 1, Column 1) to top-left.
def log_pixel_sequence_grouped(labels, rows, cols, n_colors):
    """Generates a detailed log of pixel color indices in the image.

    Reads the image from bottom-right to top-left, grouping pixels by color
    and providing column positions for each color in each row.

    Args:
        labels (numpy.ndarray): Array of color indices for each pixel.
        rows (int): Number of image rows.
        cols (int): Number of image columns.
        n_colors (int): Number of colors in the palette.
    """
    # Reshape the 1D labels array back into the 2D image structure
    labels_2d = labels.reshape(rows, cols)

    print("\n--- Pixel Color Index Sequence (Grouped by Sorted Color, Bottom-Right to Top-Left) ---")

    # Iterate through rows in reverse (bottom to top). Row Index 1 is the bottom-most row.
    for r in range(rows - 1, -1, -1):
        # 'r' is the 0-indexed row index from the top. Row label is 1-indexed from the bottom.
        row_label = f"Row Index {rows - r}"
        print(f"\n{row_label}:")

        # Iterate through the sorted color indices (0-indexed labels, 0 to N-1)
        for color_label in range(n_colors):

            # Find all column indices 'c' (from the LEFT) where the label matches 'color_label'
            column_indices = np.where(labels_2d[r, :] == color_label)[0]

            if len(column_indices) > 0:
                # Convert 0-indexed column (from LEFT) to 1-indexed column (from RIGHT)
                column_indices_from_right = cols - column_indices

                # Sort the column indices
                column_indices_from_right.sort()

                color_number = color_label + 1 # 1-indexed color number

                # Condense the list of column indices into ranges
                color_list = format_consecutive_numbers(column_indices_from_right.tolist())

                # Format: "Color X: [col1, col2, col3 ... colN]"
                print(f"Color {color_number:2}: Columns [{color_list}]")

# Expected format: python tavoosbaft.py image="/abs/path" color="Integer" rajshomar="Integer" widthcm="Integer" or widthpx="Integer"
if __name__ == '__main__':
    # Command Line Argument Parsing
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Store keys lowercased and values stripped of quotes/whitespace
            args[key.strip().lower()] = value.strip().strip('"').strip("'")
        else:
            print(f"Warning: Skipping unformatted argument: {arg}")

    # Extract and Validate Required Parameters
    image_file = args.get('image')
    color_input = args.get('color')
    rajshomar_input = args.get('rajshomar')
    widthcm_input = args.get('widthcm')
    widthpx_input = args.get('widthpx')

    # Check for missing crucial arguments
    if not image_file or not color_input or not rajshomar_input:
        print("\nFATAL ERROR: Missing required arguments (image, color, or rajshomar).")
        print("Usage Example (CM): python3 tavoosbaft.py image=\"/path/to/img.jpg\" color=\"8\" rajshomar=\"50\" widthcm=\"30\"")
        print("Usage Example (PX): python3 tavoosbaft.py image=\"/path/to/img.jpg\" color=\"8\" rajshomar=\"50\" widthpx=\"1200\"")
        sys.exit(1)

    # Check if exactly one of widthcm or widthpx is provided
    if (widthcm_input is None and widthpx_input is None) or \
       (widthcm_input is not None and widthpx_input is not None):
        print("\nFATAL ERROR: You must provide exactly one of 'widthcm' or 'widthpx' arguments.")
        sys.exit(1)

    # Setup log file and determine the log file name based on the input image
    base, _ = os.path.splitext(image_file)
    output_log_filename = f"{base}_processing.log"
    original_stdout = sys.stdout # Store original stdout

    # The entire core processing logic is wrapped in a try/finally block to ensure output redirection is restored
    try:
        # Redirect stdout to the log file using a context manager
        with open(output_log_filename, 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file

            # Start core processing (All print output goes to file)
            print(f"Starting Image Processing: {time()}")
            print(f"Input image: {image_file}")
            print(f"Target Colors (K-Means): {color_input}")

            # Data type conversion and validation
            try:
                n_colors = int(color_input)
                if n_colors <= 0:
                     raise ValueError("Color count must be a positive integer.")
            except ValueError as e:
                print(f"FATAL ERROR: Invalid value for color count: {color_input}. {e}")
                sys.exit(1)

            try:
                rajshomar = int(rajshomar_input)
                if rajshomar <= 0:
                    raise ValueError("Rajshomar value must be positive.")
            except ValueError as e:
                print(f"FATAL ERROR: Invalid value for 'rajshomar': {rajshomar_input}. {e}")
                sys.exit(1)

            # Determine target widthpx based on input mode
            target_widthpx = None

            if widthpx_input is not None:
                # Mode 1: widthpx provided
                try:
                    widthpx = float(widthpx_input)
                    if widthpx <= 0:
                         raise ValueError("widthpx value must be positive.")
                except ValueError as e:
                    print(f"FATAL ERROR: Invalid value for 'widthpx': {widthpx_input}. {e}")
                    sys.exit(1)

                target_widthpx = float(widthpx_input)

            elif widthcm_input is not None:
                # Mode 2: widthcm provided
                try:
                    widthcm = float(widthcm_input)
                    if widthcm <= 0:
                         raise ValueError("Widthcm value must be positive.")
                except ValueError as e:
                    print(f"FATAL ERROR: Invalid value for 'widthcm': {widthcm_input}. {e}")
                    sys.exit(1)

                # Covert widthcm to widthpx: widthpx = (rajshomar * widthcm) / 7
                target_widthpx = (rajshomar * widthcm) / 7
                print(f"\nTarget width calculated from widthcm: (rajshomar={rajshomar} x widthcm={widthcm}) / 7 = Target widthpx={target_widthpx:.2f} pixels.")

            # Final validation check for target_widthpx
            if target_widthpx is None or target_widthpx <= 0:
                 print(f"FATAL ERROR: Calculated or provided Target width in pixels is invalid: {target_widthpx}")
                 sys.exit(1)

            # Load image
            try:
                original_image_pil = load_image(image_file)
            except (ValueError, FileNotFoundError, IOError) as e:
                print(f"\nFATAL ERROR during image loading: {e}")
                sys.exit(1)

            # Resize image
            resized_image_array = resize_image_aspect_ratio(original_image_pil, target_widthpx)

            # Run color reduction (K-Means)
            reduced_image, labels, rows, cols = reduce_colors(resized_image_array, n_colors)

            # Log Pixel Sequence (Bottom-Right to Top-Left)
            log_pixel_sequence_grouped(labels, rows, cols, n_colors)

            # Export the Result
            base, ext = os.path.splitext(image_file)
            if not ext: ext = '.png' # Default to PNG if no extension is found

            # Construct a clear output filename including key parameters
            output_filename = f"{base}_resized_{int(target_widthpx)}_reduced_{n_colors}{ext}"

            try:
                # Save the result using Matplotlib's imsave, which handles the normalized float array
                plt.imsave(output_filename, reduced_image)
                print(f"\nSUCCESS: Color-reduced and resized image saved as: {output_filename}")
            except Exception as e:
                print(f"\nERROR: Could not save the image. Reason: {e}")

            print(f"--- Processing Finished Successfully ---")

    except Exception as e:
        # Log unexpected errors to the original stdout (terminal) for immediate visibility
        print(f"\nAN UNEXPECTED FATAL ERROR OCCURRED DURING PROCESSING: {e}")

    finally:
        # ALWAYS restore stdout to its original state (the terminal)
        sys.stdout = original_stdout

        # Print final status to the terminal
        print(f"\nProcess complete. All detailed log output was successfully written to: {output_log_filename}")
