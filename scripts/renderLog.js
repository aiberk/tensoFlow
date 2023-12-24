(function () {
  const oldLog = console.log;
  const consoleElement = document.getElementById("console");

  console.log = function (message) {
    // Call the old console.log function
    oldLog.apply(console, arguments);

    // Also display in the webpage console
    if (typeof message == "object") {
      consoleElement.innerHTML +=
        (JSON && JSON.stringify ? JSON.stringify(message) : message) + "<br />";
    } else {
      consoleElement.innerHTML += message + "<br />";
    }
  };
})();
