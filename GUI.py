import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import pytesseract
from PIL import Image, ImageTk
import subprocess
import numpy as np




pytesseract.pytesseract.tesseract_cmd = r'C:\program files\tesseract-OCR\tesseract.exe'
def browse_image():
    """Opens a file dialog to select an image."""
    global image
    filename = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.bmp")])
    if filename:
        try:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

            # Calculate aspect ratio and resize while preserving it
            height, width, _ = image.shape
            aspect_ratio = width / height
            new_width = 600  # Desired width
            new_height = int(new_width / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height))
            global original_image
            original_image = image

            print("Image loaded...")
            display_image()  # Update the displayed image
        except (IOError, cv2.error) as e:
            print(f"Error loading image: {e}")


def display_image():
    """Displays the loaded image in a designated area."""
    global image
    if image is not None:
        # Convert OpenCV image to a PIL Image for Tkinter display
        pil_image = Image.fromarray(image)
        photo_image = ImageTk.PhotoImage(pil_image)
        image_label.config(image=photo_image)
        image_label.image = photo_image  # Keep reference for garbage collection

        # Update layout after changing image
        update_layout()
















def get_grayscale():
    """Converts the original image to grayscale and displays it."""
    global image
    if image is not None and len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, image)  # In-place conversion
        print("grayscale applied...")
        display_processed_image(image)


def remove_noise():
    """Applies noise removal to the original image."""
    global image
    if image is not None:
        cv2.medianBlur(image, 5, dst=image)  # In-place modification with explicit dst
        print("Noise removed with median blur: 5")
        display_image()


def ocr_core():
    """Performs OCR on a copy of the image and displays the extracted text in a separate window."""
    global image
    if image is not None:
        # Create a copy to avoid modifying the original image
        text = pytesseract.image_to_string(image, lang='urd')
        print("text extracted....")
        print("text =", text, " ")
        # Display text in a separate window
        display_text_window(text)
        # Write the extracted text to a Notepad file
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Display a message indicating that the text has been saved
        print("Extracted text saved to extracted_text.txt")


def display_processed_image(processed_image):
    """Displays a processed image in the window."""
    global image  # Not strictly necessary here, but for clarity
    # Convert OpenCV image to PIL Image and update the labels
    pil_image = Image.fromarray(processed_image)
    photo_image = ImageTk.PhotoImage(pil_image)
    image_label.config(image=photo_image)
    image_label.image = photo_image

    # Update layout after changing image
    update_layout()


def update_layout(event=None):
    """Updates the layout to keep the text display below the image."""
    image_text_frame.update_idletasks()
    image_height = image_label.winfo_height()
    display_width = window.winfo_width() - 20  # Adjust width to account for padding/margins
    # No need to update display position anymore


def display_text_window(text):
    """Creates a separate window to display the extracted text with a copy button."""
    text_window = tk.Toplevel(window)
    text_window.title("Extracted Text")

    text_area = tk.Text(text_window, font="Noori 12")
    text_area.pack(fill=tk.BOTH, expand=True)

    text_area.insert(tk.INSERT, text)

    copy_button = ttk.Button(text_window, text="Copy",
                             command=lambda: text_window.clipboard_append(
                                 text_area.get(
                                     "1.0", tk.END)))
    open_notepad = ttk.Button(text_window, text="Open in notepad",
                              command=open_notepad_file)
    open_notepad.pack()
    copy_button.pack()


# def boxes():
#     """Performs text detection and draws bounding boxes around the detected regions.
#        Also displays the updated image with bounding boxes."""
#
#     global image  # Access the global image variable
#     if len(image.shape) == 3:
#         hImg, wImg,_= image.shape  # Get image dimensions
#     elif len(image.shape) == 2:
#         hImg, wImg = image.shape
#
#     # Extract text data using Tesseract
#     text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='urd')
#     last_column_values = list(text_data.values())[-1]
#     print(last_column_values)
#     # Filter text regions based on non-empty text and confidence threshold
#     text_regions = []
#     for i in range(len(text_data['text'])):
#         if text_data['text'][i].strip() and text_data['conf'][i] > 50:  # Adjust confidence threshold as needed
#             x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
#             text_regions.append((x, y, w, h))
#
#     # Draw bounding boxes around detected text regions
#     for box in text_regions:
#         x, y, w, h = box
#         cv2.rectangle(image, (x, y), (x + w, y + h),(255,255,0), 1)  # Draw green rectangle
#
#     # Display the image with bounding boxes
#     display_processed_image(image)  # Utilize the existing function
#     print("boxes applied...")

def boxes():
    global image
    img = image
    h, w, c = image.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def thresholding(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    thresh_img = thresholding(img);

    # dilation
    kernel = np.ones((3, 85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)


    (contours, herarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y m, h)

    img2 = img.copy()

    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (40, 100, 250), 2)

    # dilation2
    kernel = np.ones((3, 15), np.uint8)
    dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)

    img3 = img.copy()
    words_list = []

    for line in sorted_contours_lines:

        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y + w, x:x + w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        for word in sorted_contours_words:

            if cv2.contourArea(word) < 500:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
            cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 0, 0), 1)
    cv2.imshow("img2", img3)

    ninth_word = words_list[7]
    roi_9 = img[ninth_word[1]:ninth_word[3], ninth_word[0]:ninth_word[2]]
    cv2.imshow("word",roi_9)
    text = pytesseract.image_to_string(roi_9,lang='urd')
    print(text)


def print_shape():
    global image,original_image
    print("Image shape : ", len(image.shape))
    cv2.imshow("new win",original_image)


def open_notepad_file():
    """Opens the specified Notepad file using subprocess."""
    try:
        subprocess.Popen(["notepad.exe", "extracted_text.txt"])
        print("Notepad file opened successfully.")
    except FileNotFoundError:
        print("Notepad.exe not found. Please ensure it's installed.")


# Main window and UI elements
window = tk.Tk()
window.title("Urdu-OCR")
window.geometry("1000x500")

# Place the browse button at the top
browse_button = ttk.Button(window, text="Browse Image", command=browse_image)
browse_button.place(x=0, y=0)

# Create a frame for the image and text display area
#______________________________________________________

image_text_frame = tk.Frame(window, width=300)
image_text_frame.place(x=10, y=30)

# Create image labels within the frame
image_label = tk.Label(image_text_frame, bg="black")
image_label.pack()



#
# # Create the frame for the image and text
# image_text_frame = tk.Frame(window)
# image_text_frame.place(x=600, y=40, width=380, height=480)
#
# # Create the text widget
# image_label = tk.Text(image_text_frame, wrap=tk.NONE)
#
# # Pack the text widget
# image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
#_____________________________________________________________


# Create buttons within the main window
GetGray = ttk.Button(window, text="Grayscale", command=get_grayscale)
GetGray.place(x=85, y=0)

RemoveNoise = ttk.Button(window, text="Noise Removal", command=remove_noise)
RemoveNoise.place(x=162, y=0)

text_boxes_button = ttk.Button(window, text="Text Boxes", command=boxes)
text_boxes_button.place(x=320, y=0)

print_button = ttk.Button(window, text ="Print shape", command=print_shape)
print_button.place(x=390,y=0)

GetText = ttk.Button(window, text="Extract Text", command=ocr_core)
GetText.place(x=250, y=0)
# Bind the configure event to update the layout when the window is resized
window.bind('<Configure>', update_layout)

window.mainloop()
