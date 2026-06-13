(function () {
  "use strict";

  var config = window.IRIS_CONFIG || {};
  var defaultThreshold = config.defaultThreshold || "0.30";
  var state = {
    queryFile: null,
    leftFile: null,
    rightFile: null,
    busy: false
  };

  function $(id) {
    return document.getElementById(id);
  }

  function text(id, value) {
    var el = $(id);
    if (el) {
      el.textContent = value;
    }
  }

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function toast(message) {
    var box = $("toast");
    box.textContent = message;
    box.classList.add("show");
    window.setTimeout(function () {
      box.classList.remove("show");
    }, 3600);
  }

  function setLoading(message, show) {
    var loader = $("loader");
    text("loaderText", message || "Processing...");
    loader.classList.toggle("show", !!show);
    loader.setAttribute("aria-hidden", show ? "false" : "true");
  }

  function setBusy(on, message) {
    state.busy = !!on;
    var buttons = document.querySelectorAll("[data-busy-disable]");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].disabled = state.busy;
    }
    setLoading(message, state.busy);
  }

  function setDecision(kind, main, sub) {
    var el = $("decisionText");
    el.className = "decision-text " + kind;
    el.textContent = main;
    text("decisionSub", sub);
  }

  function setEnroll(kind, main, sub) {
    var el = $("enrollDecision");
    el.className = "decision-text " + kind;
    el.textContent = main;
    text("enrollSub", sub);
  }

  function asScore(value) {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    var number = Number(value);
    if (!isFinite(number)) {
      return "-";
    }
    return number.toFixed(6);
  }

  function setThreshold(value) {
    var number = parseFloat(value);
    if (!isFinite(number)) {
      number = parseFloat(defaultThreshold);
    }
    if (number < 0.10) {
      number = 0.10;
    }
    if (number > 0.95) {
      number = 0.95;
    }
    var fixed = number.toFixed(2);
    $("thresholdRange").value = fixed;
    $("thresholdText").value = fixed;
    text("thresholdLabel", fixed);
  }

  function switchTab(tabId) {
    var tabs = document.querySelectorAll(".tab-button");
    var panels = document.querySelectorAll(".panel");
    var i;
    for (i = 0; i < tabs.length; i += 1) {
      tabs[i].classList.toggle("active", tabs[i].getAttribute("data-tab") === tabId);
    }
    for (i = 0; i < panels.length; i += 1) {
      panels[i].classList.toggle("active", panels[i].id === tabId);
    }
    if (tabId === "database") {
      loadUsers();
    }
  }

  function setupTabs() {
    var tabs = document.querySelectorAll(".tab-button");
    for (var i = 0; i < tabs.length; i += 1) {
      tabs[i].addEventListener("click", function () {
        switchTab(this.getAttribute("data-tab"));
      });
    }
  }

  function setupDrop(dropId, inputId, previewId, key, after) {
    var drop = $(dropId);
    var input = $(inputId);
    var preview = $(previewId);

    function handle(file) {
      if (!file) {
        return;
      }
      state[key] = file;
      preview.src = URL.createObjectURL(file);
      drop.classList.add("has-image");
      if (after) {
        after(file);
      }
    }

    drop.addEventListener("dragover", function (event) {
      event.preventDefault();
      drop.classList.add("drag");
    });

    drop.addEventListener("dragleave", function () {
      drop.classList.remove("drag");
    });

    drop.addEventListener("drop", function (event) {
      event.preventDefault();
      drop.classList.remove("drag");
      handle(event.dataTransfer.files[0]);
    });

    input.addEventListener("change", function () {
      handle(input.files[0]);
    });
  }

  function updateHealth() {
    return fetch("/api/health", { cache: "no-store" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        text("systemStatus", data.ok ? "Online" : "Error");
        text("userCount", data.db_count);
        text("serverIp", String(data.ip || "Jetson") + ":" + String(data.port || config.serverPort || "8000"));
        if (data.threshold !== undefined && !state.busy) {
          defaultThreshold = Number(data.threshold).toFixed(2);
        }
      })
      .catch(function () {
        text("systemStatus", "Offline");
      });
  }

  function loadUsers() {
    var grid = $("usersGrid");
    grid.innerHTML = '<div class="empty-state">Loading users...</div>';
    return fetch("/api/users", { cache: "no-store" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        text("userCount", data.count || 0);
        if (!data.users || !data.users.length) {
          grid.innerHTML = '<div class="empty-state">No enrolled users yet.</div>';
          return;
        }
        var html = "";
        for (var i = 0; i < data.users.length; i += 1) {
          var user = data.users[i];
          html += '<article class="user-card">';
          html += '<div class="user-name">' + esc(user.name) + '</div>';
          html += '<span class="badge ' + (user.left ? "" : "off") + '">Left ' + (user.left ? "OK" : "Missing") + '</span>';
          html += '<span class="badge ' + (user.right ? "" : "off") + '">Right ' + (user.right ? "OK" : "Missing") + '</span>';
          html += '<div class="user-date">Created: ' + esc(user.created_at || "unknown") + '</div>';
          html += '</article>';
        }
        grid.innerHTML = html;
      })
      .catch(function () {
        grid.innerHTML = '<div class="empty-state">Could not load database.</div>';
      });
  }

  function recognize() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    if (!state.queryFile) {
      toast("Choose an iris image first.");
      return;
    }

    var fd = new FormData();
    fd.append("image", state.queryFile);
    fd.append("threshold", $("thresholdText").value || defaultThreshold || "0.30");

    setBusy(true, "Running recognition...");
    fetch("/api/recognize", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setDecision("error", "ERROR", data.error || "Recognition failed");
          toast(data.error || "Recognition failed");
          return;
        }

        text("timeMetric", String(data.elapsed_ms) + " ms");
        text("scoreMetric", asScore(data.score));
        text("eyeMetric", data.eye || "-");
        text("thresholdLabel", Number(data.threshold).toFixed(2));

        if (data.matched) {
          text("identityMetric", data.name || "-");
          setDecision("match", "MATCH FOUND", "Identity: " + data.name);
        } else {
          text("identityMetric", "Unknown");
          setDecision("nomatch", "NO MATCH", "Best score below threshold.");
        }

        if (data.top_scores && data.top_scores.length) {
          var rows = [];
          for (var i = 0; i < data.top_scores.length; i += 1) {
            var score = data.top_scores[i];
            rows.push(
              String(i + 1) + ". " + score.user + " / " + score.eye + " : " + asScore(score.score)
            );
          }
          text("scoreLog", rows.join("\n"));
        } else {
          text("scoreLog", "No users in database.");
        }
      })
      .catch(function (error) {
        setDecision("error", "ERROR", String(error));
        toast("Request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
        updateHealth();
      });
  }

  function enroll() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }

    var name = $("enrollName").value.replace(/^\s+|\s+$/g, "");
    if (!name) {
      toast("Enter user name.");
      return;
    }
    if (!state.leftFile || !state.rightFile) {
      toast("Choose both left and right eye images.");
      return;
    }

    var fd = new FormData();
    fd.append("name", name);
    fd.append("left", state.leftFile);
    fd.append("right", state.rightFile);

    setBusy(true, "Enrolling both eyes...");
    fetch("/api/register", { method: "POST", body: fd })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.ok) {
          setEnroll("error", "ERROR", data.error || "Enrollment failed");
          toast(data.error || "Enrollment failed");
          return;
        }
        setEnroll("match", "ENROLLED", "User added: " + data.name);
        text("leftTimeMetric", String(data.left_ms) + " ms");
        text("rightTimeMetric", String(data.right_ms) + " ms");
        text("enrollTimeMetric", String(data.elapsed_ms) + " ms");
        text("enrollDbMetric", String(data.db.count) + " users");
        toast("Registered both eyes for " + data.name);
        updateHealth();
        loadUsers();
      })
      .catch(function (error) {
        setEnroll("error", "ERROR", String(error));
        toast("Request failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function clearRecognition() {
    state.queryFile = null;
    $("queryInput").value = "";
    $("queryPreview").src = "";
    $("queryDrop").classList.remove("has-image");
    text("identityMetric", "-");
    text("eyeMetric", "-");
    text("scoreMetric", "-");
    text("timeMetric", "-");
    text("scoreLog", "Top matches will appear here.");
    setDecision("idle", "READY", "Choose an image to begin.");
  }

  function clearEnrollment() {
    state.leftFile = null;
    state.rightFile = null;
    $("leftInput").value = "";
    $("rightInput").value = "";
    $("leftPreview").src = "";
    $("rightPreview").src = "";
    $("leftDrop").classList.remove("has-image");
    $("rightDrop").classList.remove("has-image");
    $("enrollName").value = "";
    text("leftTimeMetric", "-");
    text("rightTimeMetric", "-");
    text("enrollTimeMetric", "-");
    text("enrollDbMetric", "-");
    setEnroll("idle", "WAITING", "Enter a name and choose both eyes.");
  }

  function deleteAll() {
    if (state.busy) {
      toast("Still processing the previous request.");
      return;
    }
    if (!window.confirm("Delete all enrolled users?")) {
      return;
    }
    setBusy(true, "Clearing database...");
    fetch("/api/delete_all", { method: "POST" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        toast(data.message || "Database cleared");
        updateHealth();
        loadUsers();
      })
      .catch(function (error) {
        toast("Delete failed: " + String(error));
      })
      .then(function () {
        setBusy(false);
      });
  }

  function init() {
    setupTabs();
    setupDrop("queryDrop", "queryInput", "queryPreview", "queryFile", function () {
      setDecision("idle", "LOADED", "Preview ready.");
      if ($("autoRecognize").checked) {
        recognize();
      }
    });
    setupDrop("leftDrop", "leftInput", "leftPreview", "leftFile", function () {
      setEnroll("idle", "LEFT READY", "Left eye loaded.");
    });
    setupDrop("rightDrop", "rightInput", "rightPreview", "rightFile", function () {
      setEnroll("idle", "RIGHT READY", "Right eye loaded.");
    });

    $("thresholdRange").addEventListener("input", function () {
      setThreshold(this.value);
    });
    $("thresholdText").addEventListener("change", function () {
      setThreshold(this.value);
    });

    $("recognizeBtn").addEventListener("click", recognize);
    $("clearQueryBtn").addEventListener("click", clearRecognition);
    $("registerBtn").addEventListener("click", enroll);
    $("clearEnrollBtn").addEventListener("click", clearEnrollment);
    $("refreshDbBtn").addEventListener("click", loadUsers);
    $("deleteDbBtn").addEventListener("click", deleteAll);

    setThreshold(defaultThreshold);
    updateHealth();
    loadUsers();
    window.setInterval(updateHealth, 7000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
