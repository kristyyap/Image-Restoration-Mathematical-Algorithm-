chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "reverseImageFilter",
    title: "Reverse Filter this Image",
    contexts: ["image"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "reverseImageFilter") {
    const imageUrl = info.srcUrl;
    const redirectUrl = `http://127.0.0.1:5500/filter.html?image=${encodeURIComponent(imageUrl)}`;
    
    chrome.tabs.create({ url: redirectUrl });
  }
});
