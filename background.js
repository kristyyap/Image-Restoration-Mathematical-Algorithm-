// Create context menu when extension is installed
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "reverseImageFilter",
    title: "Remove Image Filter",
    contexts: ["image"]
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "reverseImageFilter") {
    // Store the image URL
    chrome.storage.local.set({
      filteredImageUrl: info.srcUrl,
      fromContextMenu: true
    });
    
    // Open in a new tab instead of popup
    chrome.tabs.create({
      url: chrome.runtime.getURL('index.html')
    });
  }
});

// Handle extension icon clicks
chrome.action.onClicked.addListener((tab) => {
  // Open in a new tab when extension icon is clicked
  chrome.tabs.create({
    url: chrome.runtime.getURL('index.html')
  });
});