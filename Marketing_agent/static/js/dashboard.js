// Dashboard JavaScript Functions - Fixed Version

// Global variables
let currentUser = null;
let refreshInterval = null;
let contentData = {}; // Store content data globally

// Initialize dashboard - Updated 2025-10-30 16:30
document.addEventListener("DOMContentLoaded", function () {
  console.log('Dashboard JavaScript loaded - Version 2025-10-30 16:30');
  initializeDashboard();
  setupEventListeners();
  // Refresh stats on page load to ensure they're current
  setTimeout(() => {
    refreshStats();
  }, 1000);
});

function initializeDashboard() {
  // Add fade-in animation to cards
  const cards = document.querySelectorAll(".card");
  cards.forEach((card, index) => {
    setTimeout(() => {
      card.classList.add("fade-in");
    }, index * 100);
  });

  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
}

function setupEventListeners() {
  // Auto-refresh content every 30 seconds
  refreshInterval = setInterval(() => {
    refreshCurrentTab();
  }, 30000);

  // Handle window beforeunload
  window.addEventListener("beforeunload", function () {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  // Handle tab visibility change
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      if (refreshInterval) clearInterval(refreshInterval);
    } else {
      refreshInterval = setInterval(() => {
        refreshCurrentTab();
      }, 30000);
    }
  });
}

function refreshCurrentTab() {
  const activeTab = document.querySelector(".nav-link.active");
  if (activeTab) {
    const target = activeTab.getAttribute("data-bs-target");
    if (target) {
      const contentType = target.substring(1);
      loadContent(contentType);
    }
  }
}

// Content generation functions
function generateContent(type) {
  const button = event.target;
  const originalText = button.innerHTML;

  // Disable button and show loading
  button.disabled = true;
  button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';

  const messages = {
    blog: "Generating blog post...",
    trends: "Fetching latest trends...",
    analysis: "Running performance analysis...",
  };

  showToast("info", `${messages[type]} This may take a few moments.`);

  // Set longer timeout for generation requests
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes timeout

  fetch(`/generate/${type}`, { signal: controller.signal })
    .then((response) => {
      clearTimeout(timeoutId);
      return response.json();
    })
    .then((data) => {
      button.disabled = false;
      button.innerHTML = originalText;

      if (data.success) {
        showToast(
          "success",
          `${type.charAt(0).toUpperCase() + type.slice(1)
          } generated successfully!`
        );

        // Refresh the relevant tab
        setTimeout(() => {
          if (type === "blog") loadContent("blogs");
          else if (type === "trends") loadContent("trends");
          else if (type === "analysis") loadContent("performance");

          // Refresh stats with a longer delay to ensure file is written
          setTimeout(() => {
            refreshStats();
          }, 500);
        }, 1000);
      } else {
        showToast("error", data.error || "Generation failed");
      }
    })
    .catch((error) => {
      clearTimeout(timeoutId);
      button.disabled = false;
      button.innerHTML = originalText;

      if (error.name === "AbortError") {
        showToast("error", "Generation timed out. Please try again.");
      } else {
        showToast("error", "Network error: " + error.message);
      }
    });
}

// Content loading functions
function loadContent(type) {
  const contentDiv = document.getElementById(`${type}-content`);
  if (!contentDiv) return;

  // Show loading spinner
  contentDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 text-muted">Loading ${type}...</p>
        </div>
    `;

  fetch(`/content/${type}`)
    .then((response) => response.json())
    .then((data) => {
      // Store data globally for modal access
      contentData[type] = data;
      renderContent(type, data, contentDiv);
    })
    .catch((error) => {
      contentDiv.innerHTML = `
                <div class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                    <p>Error loading content: ${error.message}</p>
                    <button class="btn btn-outline-primary" onclick="loadContent('${type}')">
                        <i class="fas fa-redo me-2"></i>Retry
                    </button>
                </div>
            `;
    });
}

function renderContent(type, data, contentDiv) {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    renderEmptyState(type, contentDiv);
    return;
  }

  let html = "";

  switch (type) {
    case "blogs":
      html = renderBlogs(data);
      break;
    case "social":
      html = renderSocialContent(data);
      break;
    case "trends":
      html = renderTrends(data);
      break;
    case "performance":
      html = renderPerformance(data);
      break;
  }

  contentDiv.innerHTML = html;

  // Add animations
  const newCards = contentDiv.querySelectorAll(".card");
  newCards.forEach((card, index) => {
    setTimeout(() => {
      card.classList.add("slide-up");
    }, index * 50);
  });
}

function renderBlogs(data) {
  if (!Array.isArray(data)) data = [data];

  return data
    .map(
      (blog, index) => `
        <div class="card content-card mb-3">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="mb-0">${escapeHtml(blog.title)}</h6>
                <small class="text-muted">${formatDate(blog.timestamp)}</small>
            </div>
            <div class="card-body">
                <p class="card-text">${escapeHtml(
        blog.blog.substring(0, 300)
      )}...</p>
                <div class="d-flex justify-content-between align-items-center">
                    <button class="btn btn-sm btn-outline-primary" onclick="showBlogModal(${index})">
                        <i class="fas fa-eye me-1"></i>Read Full Post
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="copyBlogContent(${index})">
                        <i class="fas fa-copy me-1"></i>Copy
                    </button>
                </div>
            </div>
        </div>
    `
    )
    .join("");
}

function renderSocialContent(data) {
  let html = "";

  Object.keys(data).forEach((platform) => {
    const posts = Array.isArray(data[platform])
      ? data[platform]
      : [data[platform]];
    const platformIcon = getPlatformIcon(platform);

    posts.forEach((post, index) => {
      html += `
                <div class="card content-card mb-3">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">
                            <i class="${platformIcon} me-2"></i>
                            ${platform.charAt(0).toUpperCase() +
        platform.slice(1)
        }
                        </h6>
                        <small class="text-muted">${escapeHtml(
          post.title
        )}</small>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${escapeHtml(post.caption)}</p>
                        ${post.hashtags
          ? `
                            <div class="mt-2">
                                ${post.hashtags
            .map(
              (tag) =>
                `<span class="badge bg-primary me-1">${escapeHtml(
                  tag
                )}</span>`
            )
            .join("")}
                            </div>
                        `
          : ""
        }
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-secondary" onclick="copyToClipboard('${escapeHtml(
          post.caption
        ).replace(/'/g, "\\'")}')">
                                <i class="fas fa-copy me-1"></i>Copy Text
                            </button>
                        </div>
                    </div>
                </div>
            `;
    });
  });

  return html;
}

function renderTrends(data) {
  if (!Array.isArray(data)) return "";

  return data
    .map(
      (trend) => `
        <div class="card content-card mb-3">
            <div class="card-body">
                <h6 class="card-title">
                    <a href="${trend.url
        }" target="_blank" class="text-decoration-none">
                        ${escapeHtml(trend.title)}
                        <i class="fas fa-external-link-alt ms-1 small"></i>
                    </a>
                </h6>
                <p class="card-text text-muted">${escapeHtml(
          trend.description || "No description available"
        )}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <small class="text-muted">
                        <i class="fas fa-newspaper me-1"></i>
                        ${trend.source} | ${formatDate(trend.publishedAt)}
                    </small>
                    <span class="badge bg-info">
                        Score: ${trend.similarity_score
          ? Math.round(trend.similarity_score)
          : "N/A"
        }
                    </span>
                </div>
            </div>
        </div>
    `
    )
    .join("");
}

function renderPerformance(data) {
  if (!data || !data.summary) return "";

  const platforms = data.summary.platforms;
  let html = '<div class="row">';

  Object.keys(platforms).forEach((platform) => {
    const platformData = platforms[platform];
    const platformIcon = getPlatformIcon(platform);

    html += `
            <div class="col-md-6 mb-3">
                <div class="card content-card">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="${platformIcon} me-2"></i>
                            ${platform.charAt(0).toUpperCase() +
      platform.slice(1)
      }
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <div class="text-center">
                                    <h4 class="text-primary">${(
        platformData.avg_engagement * 100
      ).toFixed(1)}%</h4>
                                    <small class="text-muted">Avg Engagement</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center">
                                    <h4 class="text-success">${platformData.top_titles
        ? platformData.top_titles.length
        : 0
      }</h4>
                                    <small class="text-muted">Top Posts</small>
                                </div>
                            </div>
                        </div>
                        <hr>
                        <p><strong>Insights:</strong> ${escapeHtml(
        platformData.insights
      )}</p>
                        <p><strong>Recommendations:</strong> ${escapeHtml(
        platformData.recommendations
      )}</p>
                    </div>
                </div>
            </div>
        `;
  });

  html += "</div>";

  // Add global insights
  if (data.summary.global_insights) {
    const global = data.summary.global_insights;
    html += `
            <div class="card content-card mt-3">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-globe me-2"></i>Global Insights</h6>
                </div>
                <div class="card-body">
                    <p><strong>Success Factors:</strong> ${escapeHtml(
      global.common_success_factors
    )}</p>
                    <p><strong>Overall Recommendation:</strong> ${escapeHtml(
      global.overall_recommendation
    )}</p>
                    ${global.top_performing_titles
        ? `
                        <div class="mt-3">
                            <strong>Top Performing Titles:</strong>
                            <ul class="list-unstyled mt-2">
                                ${global.top_performing_titles
          .map(
            (title) =>
              `<li><i class="fas fa-star text-warning me-2"></i>${escapeHtml(
                title
              )}</li>`
          )
          .join("")}
                            </ul>
                        </div>
                    `
        : ""
      }
                </div>
            </div>
        `;
  }

  return html;
}

function renderEmptyState(type, contentDiv) {
  const emptyStates = {
    blogs: {
      icon: "fas fa-blog",
      message: "No blog posts generated yet.",
      action: 'Click "Generate Blog" to create your first post.',
    },
    social: {
      icon: "fas fa-share-alt",
      message: "No social media content generated yet.",
      action: 'Click "Generate Social" to create posts for all platforms.',
    },
    trends: {
      icon: "fas fa-chart-line",
      message: "No trends data available.",
      action: 'Click "Fetch Trends" to get the latest industry insights.',
    },
    performance: {
      icon: "fas fa-analytics",
      message: "No performance data available.",
      action: 'Click "Run Analysis" to generate insights.',
    },
  };

  const state = emptyStates[type];
  contentDiv.innerHTML = `
        <div class="text-center text-muted py-5">
            <i class="${state.icon} fa-4x mb-3"></i>
            <h5>${state.message}</h5>
            <p>${state.action}</p>
        </div>
    `;
}

// Blog-specific functions
function showBlogModal(index) {
  const blogs = contentData.blogs;
  if (!blogs || !blogs[index]) return;

  const blog = blogs[index];
  const modal = new bootstrap.Modal(
    document.getElementById("contentModal") || createContentModal()
  );
  document.getElementById("contentModalTitle").textContent = blog.title;
  document.getElementById("contentModalBody").innerHTML = blog.blog.replace(
    /\n/g,
    "<br>"
  );
  modal.show();
}

function copyBlogContent(index) {
  const blogs = contentData.blogs;
  if (!blogs || !blogs[index]) return;

  const blog = blogs[index];
  copyToClipboard(blog.blog);
}

// Utility functions
function getPlatformIcon(platform) {
  const icons = {
    linkedin: "fab fa-linkedin",
    twitter: "fab fa-twitter",
    youtube: "fab fa-youtube",
    facebook: "fab fa-facebook",
    instagram: "fab fa-instagram",
  };
  return icons[platform] || "fas fa-share-alt";
}

function formatDate(dateString) {
  if (!dateString) return "Unknown date";
  try {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateString;
  }
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function createContentModal() {
  const modalHtml = `
        <div class="modal fade" id="contentModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg modal-dialog-scrollable">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="contentModalTitle"></h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="contentModalBody" style="max-height: 70vh; overflow-y: auto;"></div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="copyModalContent()">
                            <i class="fas fa-copy me-1"></i>Copy Content
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
  document.body.insertAdjacentHTML("beforeend", modalHtml);
  return document.getElementById("contentModal");
}

function copyModalContent() {
  const content = document.getElementById("contentModalBody").textContent;
  copyToClipboard(content);
}

function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      showToast("success", "Content copied to clipboard!");
    })
    .catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      showToast("success", "Content copied to clipboard!");
    });
}

function exportContent() {
  showToast("info", "Preparing export...");

  fetch("/export")
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        showToast("error", data.error);
      } else {
        // Download the data as JSON
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `marketing_content_${new Date().toISOString().split("T")[0]
          }.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("success", "Content exported successfully!");
      }
    })
    .catch((error) => {
      showToast("error", "Export failed: " + error.message);
    });
}

function refreshStats() {
  // Refresh stats using dedicated endpoint
  console.log("🔄 Refreshing stats...");
  fetch('/stats')
    .then((response) => response.json())
    .then((stats) => {
      console.log("📊 Received stats:", stats);
      // Update stat cards
      const statCards = document.querySelectorAll(".card h4");
      if (statCards.length >= 3) {
        console.log("📈 Updating UI - Blogs:", stats.blogs, "Social:", stats.social_posts, "Trends:", stats.trends);
        statCards[0].textContent = stats.blogs || 0;
        statCards[1].textContent = stats.social_posts || 0;
        statCards[2].textContent = stats.trends || 0;
      }
    })
    .catch((error) => {
      console.error("Error refreshing stats:", error);
    });
}

// Toast notification system
function showToast(type, message) {
  const toastContainer =
    document.getElementById("toast-container") || createToastContainer();

  const toastId = "toast-" + Date.now();
  const toastHtml = `
        <div class="toast align-items-center text-white bg-${type === "error" ? "danger" : type === "success" ? "success" : "info"
    } border-0" role="alert" id="${toastId}">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${type === "error"
      ? "exclamation-triangle"
      : type === "success"
        ? "check-circle"
        : "info-circle"
    } me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

  toastContainer.insertAdjacentHTML("beforeend", toastHtml);

  const toast = new bootstrap.Toast(document.getElementById(toastId), {
    autohide: true,
    delay: type === "error" ? 5000 : 3000,
  });

  toast.show();

  // Remove toast element after it's hidden
  document
    .getElementById(toastId)
    .addEventListener("hidden.bs.toast", function () {
      this.remove();
    });
}

function createToastContainer() {
  const containerHtml = `
        <div class="toast-container position-fixed bottom-0 end-0 p-3" id="toast-container"></div>
    `;
  document.body.insertAdjacentHTML("beforeend", containerHtml);
  return document.getElementById("toast-container");
}
// Dashboard Topic Selection Functions
function showTopicSelection() {
  console.log('showTopicSelection called - showing on dashboard');
  
  // Show the topic selection section
  const topicSection = document.getElementById('topicSelectionSection');
  topicSection.classList.remove('d-none');
  
  // Scroll to the section
  topicSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  
  // Show loading state
  document.getElementById('topicLoading').classList.remove('d-none');
  document.getElementById('topicSelectionContent').classList.add('d-none');
  
  // Load topics
  fetch('/generate_topics')
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        const topicList = document.getElementById('topicList');
        topicList.innerHTML = ''; // Clear existing content
        
        data.topics.forEach((topic, index) => {
          const relatedCount = topic.related_news ? topic.related_news.length : 0;
          topicList.innerHTML += `
            <div class="card mb-3 topic-card" style="cursor: pointer;" onclick="selectTopic(${index})">
              <div class="card-body">
                <div class="form-check">
                  <input class="form-check-input" type="radio" name="dashboardTopic" value="${index}" id="dashboardTopic${index}">
                  <label class="form-check-label w-100" for="dashboardTopic${index}">
                    <h6 class="mb-1">${escapeHtml(topic.title)}</h6>
                    <small class="text-muted">
                      <i class="fas fa-newspaper me-1"></i>${relatedCount} related articles
                    </small>
                  </label>
                </div>
              </div>
            </div>
          `;
        });
        
        // Show content and hide loading
        document.getElementById('topicLoading').classList.add('d-none');
        document.getElementById('topicSelectionContent').classList.remove('d-none');
        
      } else {
        showToast('error', data.error || 'Failed to load topics');
        hideTopicSelection();
      }
    })
    .catch(error => {
      showToast('error', 'Failed to load topics: ' + error.message);
      hideTopicSelection();
    });
}

function selectTopic(index) {
  // Select the radio button
  const radio = document.getElementById(`dashboardTopic${index}`);
  radio.checked = true;
  
  // Remove selected class from all cards
  document.querySelectorAll('.topic-card').forEach(card => {
    card.classList.remove('border-success', 'bg-light');
  });
  
  // Add selected class to clicked card
  const selectedCard = radio.closest('.topic-card');
  selectedCard.classList.add('border-success', 'bg-light');
  
  // Enable generate button
  document.getElementById('generateSocialBtn').disabled = false;
}

function hideTopicSelection() {
  const topicSection = document.getElementById('topicSelectionSection');
  topicSection.classList.add('d-none');
  
  // Reset form
  document.querySelectorAll('input[name="dashboardTopic"]').forEach(radio => {
    radio.checked = false;
  });
  document.getElementById('generateSocialBtn').disabled = true;
  
  // Remove selected styling
  document.querySelectorAll('.topic-card').forEach(card => {
    card.classList.remove('border-success', 'bg-light');
  });
}

function generateSocialFromDashboard() {
  const selectedTopic = document.querySelector('input[name="dashboardTopic"]:checked');
  const tone = document.getElementById('toneSelect').value;
  const audience = document.getElementById('audienceSelect').value;
  
  if (!selectedTopic) {
    showToast('error', 'Please select a topic');
    return;
  }
  
  // Show loading state
  const generateBtn = document.getElementById('generateSocialBtn');
  const originalText = generateBtn.innerHTML;
  generateBtn.disabled = true;
  generateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
  
  showToast('info', 'Generating social content... This may take a few moments.');
  
  // Generate content with selections
  fetch('/generate_social_with_selection', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topic_index: parseInt(selectedTopic.value),
      tone: tone,
      audience: audience
    })
  })
    .then(response => response.json())
    .then(data => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;
      
      if (data.success) {
        showToast('success', 'Social content generated successfully!');
        
        // Hide topic selection
        hideTopicSelection();
        
        // Switch to social tab and refresh content
        const socialTab = document.getElementById('social-tab');
        const socialTabInstance = new bootstrap.Tab(socialTab);
        socialTabInstance.show();
        
        setTimeout(() => {
          loadContent('social');
          refreshStats();
        }, 1000);
      } else {
        showToast('error', data.error || 'Failed to generate content');
      }
    })
    .catch(error => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;
      showToast('error', 'Failed to generate content: ' + error.message);
    });
}