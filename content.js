// Content script to handle image context menu interactions
// This file helps with any additional image processing if needed

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getImageData") {
    // If we need to get image data from the current page
    const img = document.querySelector(`img[src="${request.imageUrl}"]`);
    if (img) {
      sendResponse({
        success: true,
        imageData: {
          src: img.src,
          alt: img.alt || "Selected Image",
          width: img.naturalWidth,
          height: img.naturalHeight
        }
      });
    } else {
      sendResponse({ success: false });
    }
  }
});