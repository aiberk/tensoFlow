document.addEventListener("DOMContentLoaded", function () {
  // Function to extract folder name from URL
  function getFolderNameFromUrl() {
    const url = window.location.href;
    const urlSegments = url.split("/");
    // Assuming the folder is the second last segment in the URL
    const folderName = urlSegments[urlSegments.length - 2];
    return folderName;
  }

  function camelCaseToNormalCase(str) {
    // Add a space before all caps and trim
    return (
      str
        .replace(/([A-Z])/g, " $1")
        .trim()
        // Capitalize the first letter
        .replace(/^./, function (str) {
          return str.toUpperCase();
        })
    );
  }

  // Set the title based on the folder name
  const folderName = getFolderNameFromUrl();
  const folderNameEdit = camelCaseToNormalCase(folderName);
  const title = document.getElementById("title");

  title.innerText = folderNameEdit;
  document.title = folderNameEdit;
});
