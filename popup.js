// Elements
const originalUpload = document.getElementById("originalUpload");
const originalPreview = document.getElementById("originalPreview");
const filteredPreview = document.getElementById("filteredPreview");
const filteredStatus = document.getElementById("filteredStatus");
const processButton = document.getElementById("processButton");
const uploadSection = document.getElementById("uploadSection");
const processSection = document.getElementById("processSection");
const originalImage = document.getElementById("originalImage");
const filteredImage = document.getElementById("filteredImage");
const restoredImage = document.getElementById("restoredImage");
const similarityBar = document.getElementById("similarityBar");
const similarityScore = document.getElementById("similarityScore");
const backContainer = document.getElementById("backContainer");
const backToUpload = document.getElementById("backToUpload");
const breadcrumbProcess = document.getElementById("breadcrumb-process");
const loadingIndicator = document.getElementById("loadingIndicator");

let originalFile = null;
let filteredImageUrl = null;
let originalImageData = null;

// Check for stored filtered image from context menu
window.addEventListener("DOMContentLoaded", function () {
  chrome.storage.local.get(
    ["filteredImageUrl", "fromContextMenu"],
    (result) => {
      if (result.filteredImageUrl && result.fromContextMenu) {
        filteredImageUrl = result.filteredImageUrl;
        displayFilteredImage(filteredImageUrl);
        // Clear the storage flag
        chrome.storage.local.remove(["fromContextMenu"]);
      }
    }
  );
});

function displayFilteredImage(imageUrl) {
  filteredStatus.textContent = " ";
  filteredStatus.style.color = "#4CAF50";

  filteredPreview.innerHTML = `<img src="${imageUrl}" alt="Filtered Image" style="max-width: 100%; max-height: 150px; border-radius: 5px;">`;
  filteredImage.innerHTML = `<img src="${imageUrl}" alt="Filtered Image" style="max-width: 100%; max-height: 200px;">`;

  checkProcessButton();
}

// Upload preview for original image
originalUpload.addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (file) {
    originalFile = file;
    const reader = new FileReader();
    reader.onload = function (e) {
      originalImageData = e.target.result;
      originalPreview.innerHTML = `<img src="${originalImageData}" alt="Original Image" style="max-width: 100%; max-height: 150px; border-radius: 5px; border: 1px solid #ccc; margin-top: 8px;">`;
      originalImage.innerHTML = `<img src="${originalImageData}" alt="Original Image" style="max-width: 100%; max-height: 200px;">`;
      checkProcessButton();
    };
    reader.readAsDataURL(file);
  }
});

function checkProcessButton() {
  if (originalFile && filteredImageUrl) {
    processButton.style.display = "block";
  }
}

// Convert URL image to File object for upload
async function urlToFile(url, filename) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type });
  } catch (error) {
    console.error("Error converting URL to file:", error);
    throw error;
  }
}

// Main process button
processButton.addEventListener("click", async function () {
  if (!originalFile || !filteredImageUrl) return;

  loadingIndicator.classList.remove("hidden");
  processButton.disabled = true;

  // Hide upload, show process/results UI
  uploadSection.classList.add("hidden");
  processSection.classList.remove("hidden");
  backContainer.classList.remove("hidden");
  breadcrumbProcess.classList.remove("hidden");

  try {
    // Convert filtered image URL to File
    const filteredFile = await urlToFile(
      filteredImageUrl,
      "filtered_image.png"
    );

    // Prepare the form data
    const formData = new FormData();
    formData.append("original", originalFile);
    formData.append("filtered", filteredFile);

    // Replace with your actual server URL
    const serverUrl = "http://localhost:5000/process"; // Change this to your server URL

    const response = await fetch(serverUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();

    loadingIndicator.classList.add("hidden");
    restoredImage.innerHTML = `<img src="${data.restored}" alt="Restored Image" style="max-width: 100%; max-height: 200px;">`;

    const similarity = data.similarity || 0;
    similarityBar.style.width = similarity + "%";
    similarityScore.textContent = similarity + "%";

    // Update similarity bar color
    similarityBar.classList.remove("high", "medium", "low");
    if (similarity >= 75) similarityBar.classList.add("high");
    else if (similarity >= 35) similarityBar.classList.add("medium");
    else similarityBar.classList.add("low");

    processButton.disabled = false;
  } catch (error) {
    loadingIndicator.classList.add("hidden");
    alert(
      "Failed to process image: " +
        error.message +
        "\n\nMake sure your server is running on the correct URL."
    );
    processButton.disabled = false;
  }
});

// Back to Upload Section
backToUpload.addEventListener("click", function (e) {
  e.preventDefault();
  uploadSection.classList.remove("hidden");
  processSection.classList.add("hidden");
  backContainer.classList.add("hidden");
  breadcrumbProcess.classList.add("hidden");

  // Reset original image upload
  originalUpload.value = "";
  originalPreview.innerHTML = "";
  originalImage.innerHTML = "";
  originalFile = null;
  originalImageData = null;

  // Keep filtered image data
  restoredImage.innerHTML = "";
  similarityBar.style.width = "0%";
  similarityScore.textContent = "0%";
  processButton.style.display = "none";

  checkProcessButton();
});
