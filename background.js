chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "reverse-image-filter",
    title: "Reverse Image Filter",
    contexts: ["image"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "reverse-image-filter" && info.srcUrl) {
    // 替换为你 Web 系统的实际 URL
    const targetUrl = `http://localhost:5000/?filtered=${encodeURIComponent(
      info.srcUrl
    )}`;
    chrome.tabs.create({ url: targetUrl });
  }
});
