// Dashboard JavaScript Functions - Fixed Version

// Global variables
let currentUser = null;
let refreshInterval = null;
let contentData = {}; // Store content data globally

// Initialize dashboard - Updated 2025-11-04 16:30
document.addEventListener("DOMContentLoaded", function () {
  console.log("Dashboard JavaScript loaded - Version 2025-11-04 16:30");
  initializeDashboard();
  setupEventListeners();

  // Load initial content
  setTimeout(() => {
    loadContent("blogs");
  }, 500);

  // Add event listeners for main tab clicks
  document
    .querySelectorAll('#contentTabs [data-bs-toggle="tab"]')
    .forEach((tab) => {
      tab.addEventListener("shown.bs.tab", function (e) {
        const target = e.target.getAttribute("data-bs-target").substring(1);
        console.log("Main tab clicked:", target);
        if (target === "social") {
          // Load LinkedIn content by default when social tab is opened
          loadSocialContent("linkedin");
        } else {
          loadContent(target);
        }
      });
    });

  // Add event listeners for social sub-tab clicks
  document
    .querySelectorAll('#socialTabs [data-bs-toggle="tab"]')
    .forEach((tab) => {
      tab.addEventListener("shown.bs.tab", function (e) {
        const target = e.target.getAttribute("data-bs-target").substring(1);
        console.log("Social sub-tab clicked:", target);
        loadSocialContent(target);
      });
    });

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
  // This function is kept for backward compatibility with other buttons
  // Blog generation now uses showBlogTopicSelection() instead
  if (type === "blog") {
    showBlogTopicSelection();
    return;
  }

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
          `${
            type.charAt(0).toUpperCase() + type.slice(1)
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
    .map((blog, index) => {
      // Handle case where blog content might be JSON string
      let blogContent = blog.blog;
      if (typeof blogContent === "string" && blogContent.startsWith("{")) {
        try {
          const parsed = JSON.parse(blogContent);
          blogContent = parsed.blog || blogContent;
        } catch (e) {
          // If parsing fails, use original content
        }
      }

      return `
        <div class="card content-card mb-3">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="mb-0">${escapeHtml(blog.title)}</h6>
                <small class="text-muted">${formatDate(blog.timestamp)}</small>
            </div>
            <div class="card-body">
                <p class="card-text">${escapeHtml(
                  blogContent.substring(0, 300)
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
    `;
    })
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
                            ${
                              platform.charAt(0).toUpperCase() +
                              platform.slice(1)
                            }
                        </h6>
                        <small class="text-muted">
                            ${escapeHtml(post.title)}
                            ${
                              post.timestamp
                                ? ` | ${formatDate(post.timestamp)}`
                                : ""
                            }
                        </small>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${escapeHtml(post.caption)}</p>
                        ${
                          post.hashtags
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

function loadSocialContent(platform) {
  const contentDiv = document.getElementById(`${platform}-content`);

  // Show loading spinner
  contentDiv.innerHTML = `
    <div class="text-center">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
      <p class="mt-2 text-muted">Loading ${platform} content...</p>
    </div>
  `;

  fetch("/content/social")
    .then((response) => response.json())
    .then((data) => {
      if (!data || !data[platform]) {
        contentDiv.innerHTML = `
          <div class="text-center text-muted">
            <i class="fab fa-${platform} fa-3x mb-3"></i>
            <p>No ${
              platform.charAt(0).toUpperCase() + platform.slice(1)
            } content generated yet.</p>
          </div>
        `;
        return;
      }

      const posts = Array.isArray(data[platform])
        ? data[platform]
        : [data[platform]];
      let html = "";

      posts.forEach((post) => {
        html += `
          <div class="card content-card mb-3">
            <div class="card-header d-flex justify-content-between align-items-center">
              <h6 class="mb-0">${escapeHtml(post.title)}</h6>
              <small class="text-muted">
                <i class="fab fa-${platform} me-1"></i>
                ${platform.charAt(0).toUpperCase() + platform.slice(1)}
                ${post.timestamp ? ` | ${formatDate(post.timestamp)}` : ""}
              </small>
            </div>
            <div class="card-body">
              <p class="card-text">${escapeHtml(post.caption)}</p>
              ${
                post.hashtags
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
              ${
                post.script_intro
                  ? `
                <div class="mt-3">
                  <h6>Script Intro:</h6>
                  <p class="text-muted">${escapeHtml(post.script_intro)}</p>
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
                ${
                  post.script_intro
                    ? `
                  <button class="btn btn-sm btn-outline-info ms-2" onclick="copyToClipboard('${escapeHtml(
                    post.script_intro
                  ).replace(/'/g, "\\'")}')">
                    <i class="fas fa-copy me-1"></i>Copy Script
                  </button>
                `
                    : ""
                }
              </div>
            </div>
          </div>
        `;
      });

      contentDiv.innerHTML = html;
    })
    .catch((error) => {
      contentDiv.innerHTML = `
        <div class="text-center text-danger">
          <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
          <p>Error loading ${platform} content: ${error.message}</p>
          <button class="btn btn-outline-primary" onclick="loadSocialContent('${platform}')">
            <i class="fas fa-redo me-2"></i>Retry
          </button>
        </div>
      `;
    });
}

function renderTrends(data) {
  if (!Array.isArray(data)) return "";

  // Sort trends by similarity_score in descending order (highest to lowest)
  const sortedData = data.sort((a, b) => {
    const scoreA = a.similarity_score || 0;
    const scoreB = b.similarity_score || 0;
    return scoreB - scoreA; // Descending order
  });

  return sortedData
    .map(
      (trend) => `
        <div class="card content-card mb-3">
            <div class="card-body">
                <h6 class="card-title">
                    <a href="${
                      trend.url
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
                        Score: ${
                          trend.similarity_score
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
                            ${
                              platform.charAt(0).toUpperCase() +
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
                                    <h4 class="text-success">${
                                      platformData.top_titles
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
                    ${
                      global.top_performing_titles
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
  if (!blogs || !blogs[index]) {
    showToast("error", "Blog content not found. Please refresh the page.");
    return;
  }

  const blog = blogs[index];
  let blogContent = blog.blog;

  // Handle case where blog content might be JSON string
  if (typeof blogContent === "string" && blogContent.startsWith("{")) {
    try {
      const parsed = JSON.parse(blogContent);
      blogContent = parsed.blog || blogContent;
    } catch (e) {
      // If parsing fails, use original content
    }
  }

  // Store blog data globally for image prompt generation
  currentBlogData = {
    title: blog.title,
    blog: blogContent,
    outline: blog.outline || [],
    industry: blog.industry || "",
    tone: blog.tone || "",
    audience: blog.audience || "",
  };

  const modal = new bootstrap.Modal(
    document.getElementById("contentModal") || createContentModal()
  );
  document.getElementById("contentModalTitle").textContent = blog.title;
  document.getElementById("contentModalBody").innerHTML = blogContent.replace(
    /\n/g,
    "<br>"
  );

  // Show the image prompt button for blogs
  showImagePromptButton();

  modal.show();
}

function copyBlogContent(index) {
  const blogs = contentData.blogs;
  if (!blogs || !blogs[index]) return;

  const blog = blogs[index];
  let blogContent = blog.blog;

  // Handle case where blog content might be JSON string
  if (typeof blogContent === "string" && blogContent.startsWith("{")) {
    try {
      const parsed = JSON.parse(blogContent);
      blogContent = parsed.blog || blogContent;
    } catch (e) {
      // If parsing fails, use original content
    }
  }

  copyToClipboard(blogContent);
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
        a.download = `marketing_content_${
          new Date().toISOString().split("T")[0]
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
  fetch("/stats")
    .then((response) => response.json())
    .then((stats) => {
      console.log("📊 Received stats:", stats);
      // Update stat cards
      const statCards = document.querySelectorAll(".card h4");
      if (statCards.length >= 3) {
        console.log(
          "📈 Updating UI - Blogs:",
          stats.blogs,
          "Social:",
          stats.social_posts,
          "Trends:",
          stats.trends
        );
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
        <div class="toast align-items-center text-white bg-${
          type === "error" ? "danger" : type === "success" ? "success" : "info"
        } border-0" role="alert" id="${toastId}">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${
                      type === "error"
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
  console.log("showTopicSelection called - showing on dashboard");

  // Show the topic selection section
  const topicSection = document.getElementById("topicSelectionSection");
  topicSection.classList.remove("d-none");

  // Scroll to the section
  topicSection.scrollIntoView({ behavior: "smooth", block: "start" });

  // Show loading state
  document.getElementById("topicLoading").classList.remove("d-none");
  document.getElementById("topicSelectionContent").classList.add("d-none");

  // Load topics
  fetch("/generate_topics")
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        const topicList = document.getElementById("topicList");
        topicList.innerHTML = ""; // Clear existing content

        // Debug: Log the topics we received
        console.log("📋 Received topics:", data.topics);

        data.topics.forEach((topic, index) => {
          const relatedCount = topic.related_news
            ? topic.related_news.length
            : 0;
          console.log(`📝 Topic ${index}: ${topic.title}`);

          const relevanceScore = topic.relevance_score || 0;
          topicList.innerHTML += `
            <div class="card mb-3 topic-card" style="cursor: pointer;" onclick="selectTopic(${index})">
              <div class="card-body">
                <div class="form-check">
                  <input class="form-check-input" type="radio" name="dashboardTopic" value="${index}" id="dashboardTopic${index}">
                  <label class="form-check-label w-100" for="dashboardTopic${index}">
                    <div class="d-flex justify-content-between align-items-start">
                      <div>
                        <h6 class="mb-1">${escapeHtml(topic.title)}</h6>
                        <small class="text-muted">
                          <i class="fas fa-newspaper me-1"></i>${relatedCount} related articles
                        </small>
                      </div>
                      <span class="badge bg-info">
                        Score: ${Math.round(relevanceScore)}
                      </span>
                    </div>
                  </label>
                </div>
              </div>
            </div>
          `;
        });

        // Set default industry based on current niche (if available)
        setDefaultIndustry();

        // Show content and hide loading
        document.getElementById("topicLoading").classList.add("d-none");
        document
          .getElementById("topicSelectionContent")
          .classList.remove("d-none");
      } else {
        showToast("error", data.error || "Failed to load topics");
        hideTopicSelection();
      }
    })
    .catch((error) => {
      showToast("error", "Failed to load topics: " + error.message);
      hideTopicSelection();
    });
}

function selectTopic(index) {
  console.log(`🎯 User selected topic index: ${index}`);

  // Select the radio button
  const radio = document.getElementById(`dashboardTopic${index}`);
  radio.checked = true;

  // Debug: Log the selected topic details
  const topicTitle = radio
    .closest(".topic-card")
    .querySelector("h6").textContent;
  console.log(`📝 Selected topic title: "${topicTitle}"`);

  // Remove selected class from all cards
  document.querySelectorAll(".topic-card").forEach((card) => {
    card.classList.remove("border-success", "bg-light");
  });

  // Add selected class to clicked card
  const selectedCard = radio.closest(".topic-card");
  selectedCard.classList.add("border-success", "bg-light");

  // Enable generate button
  document.getElementById("generateSocialBtn").disabled = false;
}

function hideTopicSelection() {
  const topicSection = document.getElementById("topicSelectionSection");
  topicSection.classList.add("d-none");

  // Reset form
  document.querySelectorAll('input[name="dashboardTopic"]').forEach((radio) => {
    radio.checked = false;
  });
  document.getElementById("generateSocialBtn").disabled = true;

  // Remove selected styling
  document.querySelectorAll(".topic-card").forEach((card) => {
    card.classList.remove("border-success", "bg-light");
  });
}

function generateSocialFromDashboard() {
  const selectedTopic = document.querySelector(
    'input[name="dashboardTopic"]:checked'
  );
  const industry = document.getElementById("industrySelect").value;
  const tone = document.getElementById("toneSelect").value;
  const audience = document.getElementById("audienceSelect").value;

  if (!selectedTopic) {
    showToast("error", "Please select a topic");
    return;
  }

  // Show loading state
  const generateBtn = document.getElementById("generateSocialBtn");
  const originalText = generateBtn.innerHTML;
  generateBtn.disabled = true;
  generateBtn.innerHTML =
    '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';

  showToast(
    "info",
    `Generating social content for ${industry}... This may take a few moments.`
  );

  // Debug: Log what we're sending
  const requestData = {
    topic_index: parseInt(selectedTopic.value),
    industry: industry,
    tone: tone,
    audience: audience,
  };

  console.log("🚀 Sending request:", requestData);
  console.log("📝 Selected topic element:", selectedTopic);
  console.log("🔢 Topic index:", selectedTopic.value);

  // Generate content with selections
  fetch("/generate_social_with_selection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestData),
  })
    .then((response) => response.json())
    .then((data) => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;

      if (data.success) {
        showToast("success", "Social content generated successfully!");

        // Hide topic selection
        hideTopicSelection();

        // Switch to social tab and refresh content
        const socialTab = document.getElementById("social-tab");
        const socialTabInstance = new bootstrap.Tab(socialTab);
        socialTabInstance.show();

        setTimeout(() => {
          // Load LinkedIn content by default
          loadSocialContent("linkedin");
          refreshStats();
        }, 1000);
      } else {
        showToast("error", data.error || "Failed to generate content");
      }
    })
    .catch((error) => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;
      showToast("error", "Failed to generate content: " + error.message);
    });
}

// Set default industry based on current business context
function setDefaultIndustry() {
  // Map common business contexts to industry options
  const industryMappings = {
    saas: "IT & Dev",
    software: "IT & Dev",
    technology: "IT & Dev",
    fintech: "Fintech",
    finance: "Fintech",
    healthcare: "Healthcare",
    education: "Education",
    retail: "Retail",
    ecommerce: "Retail",
    logistics: "Logistics",
    "real estate": "Real Estate",
    marketing: "Sales/Marketing",
    sales: "Sales/Marketing",
    hr: "HRTech",
    legal: "Legal",
    media: "Media",
    travel: "Travel",
    energy: "Energy",
    agriculture: "Agritech",
    government: "Government",
  };

  // Try to detect industry from page content or use default
  const pageText = document.body.textContent.toLowerCase();
  let detectedIndustry = "IT & Dev"; // Default

  for (const [keyword, industry] of Object.entries(industryMappings)) {
    if (pageText.includes(keyword)) {
      detectedIndustry = industry;
      break;
    }
  }

  // Set the detected industry as selected for both social and blog
  const industrySelect = document.getElementById("industrySelect");
  if (industrySelect) {
    industrySelect.value = detectedIndustry;
  }

  const blogIndustrySelect = document.getElementById("blogIndustrySelect");
  if (blogIndustrySelect) {
    blogIndustrySelect.value = detectedIndustry;
  }
}

// Blog Topic Selection Functions
function showBlogTopicSelection() {
  console.log("showBlogTopicSelection called - showing on dashboard");

  // Hide social topic selection if it's open
  const socialTopicSection = document.getElementById("topicSelectionSection");
  socialTopicSection.classList.add("d-none");

  // Show the blog topic selection section
  const blogTopicSection = document.getElementById("blogTopicSelectionSection");
  blogTopicSection.classList.remove("d-none");

  // Scroll to the section
  blogTopicSection.scrollIntoView({ behavior: "smooth", block: "start" });

  // Show loading state
  document.getElementById("blogTopicLoading").classList.remove("d-none");
  document.getElementById("blogTopicSelectionContent").classList.add("d-none");

  // Load topics
  fetch("/generate_topics")
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        const topicList = document.getElementById("blogTopicList");
        topicList.innerHTML = ""; // Clear existing content

        // Debug: Log the topics we received
        console.log("📋 Received blog topics:", data.topics);

        data.topics.forEach((topic, index) => {
          const relatedCount = topic.related_news
            ? topic.related_news.length
            : 0;
          console.log(`📝 Blog Topic ${index}: ${topic.title}`);

          const relevanceScore = topic.relevance_score || 0;
          topicList.innerHTML += `
            <div class="card mb-3 blog-topic-card" style="cursor: pointer;" onclick="selectBlogTopic(${index})">
              <div class="card-body">
                <div class="form-check">
                  <input class="form-check-input" type="radio" name="dashboardBlogTopic" value="${index}" id="dashboardBlogTopic${index}">
                  <label class="form-check-label w-100" for="dashboardBlogTopic${index}">
                    <div class="d-flex justify-content-between align-items-start">
                      <div>
                        <h6 class="mb-1">${escapeHtml(topic.title)}</h6>
                        <small class="text-muted">
                          <i class="fas fa-newspaper me-1"></i>${relatedCount} related articles
                        </small>
                      </div>
                      <span class="badge bg-info">
                        Score: ${Math.round(relevanceScore)}
                      </span>
                    </div>
                  </label>
                </div>
              </div>
            </div>
          `;
        });

        // Set default industry based on current business context
        setDefaultIndustry();

        // Show content and hide loading
        document.getElementById("blogTopicLoading").classList.add("d-none");
        document
          .getElementById("blogTopicSelectionContent")
          .classList.remove("d-none");
      } else {
        showToast("error", data.error || "Failed to load topics");
        hideBlogTopicSelection();
      }
    })
    .catch((error) => {
      showToast("error", "Failed to load topics: " + error.message);
      hideBlogTopicSelection();
    });
}

function selectBlogTopic(index) {
  console.log(`🎯 User selected blog topic index: ${index}`);

  // Select the radio button
  const radio = document.getElementById(`dashboardBlogTopic${index}`);
  radio.checked = true;

  // Debug: Log the selected topic details
  const topicTitle = radio
    .closest(".blog-topic-card")
    .querySelector("h6").textContent;
  console.log(`📝 Selected blog topic title: "${topicTitle}"`);

  // Remove selected class from all cards
  document.querySelectorAll(".blog-topic-card").forEach((card) => {
    card.classList.remove("border-primary", "bg-light");
  });

  // Add selected class to clicked card
  const selectedCard = radio.closest(".blog-topic-card");
  selectedCard.classList.add("border-primary", "bg-light");

  // Enable generate button
  document.getElementById("generateBlogBtn").disabled = false;
}

function hideBlogTopicSelection() {
  const blogTopicSection = document.getElementById("blogTopicSelectionSection");
  blogTopicSection.classList.add("d-none");

  // Reset form
  document
    .querySelectorAll('input[name="dashboardBlogTopic"]')
    .forEach((radio) => {
      radio.checked = false;
    });
  document.getElementById("generateBlogBtn").disabled = true;

  // Remove selected styling
  document.querySelectorAll(".blog-topic-card").forEach((card) => {
    card.classList.remove("border-primary", "bg-light");
  });
}

function generateBlogFromDashboard() {
  const selectedTopic = document.querySelector(
    'input[name="dashboardBlogTopic"]:checked'
  );
  const industry = document.getElementById("blogIndustrySelect").value;
  const tone = document.getElementById("blogToneSelect").value;
  const audience = document.getElementById("blogAudienceSelect").value;

  if (!selectedTopic) {
    showToast("error", "Please select a topic");
    return;
  }

  // Show loading state
  const generateBtn = document.getElementById("generateBlogBtn");
  const originalText = generateBtn.innerHTML;
  generateBtn.disabled = true;
  generateBtn.innerHTML =
    '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';

  showToast(
    "info",
    `Generating blog content for ${industry}... This may take a few moments.`
  );

  // Debug: Log what we're sending
  const requestData = {
    topic_index: parseInt(selectedTopic.value),
    industry: industry,
    tone: tone,
    audience: audience,
  };

  console.log("🚀 Sending blog request:", requestData);
  console.log("📝 Selected blog topic element:", selectedTopic);
  console.log("🔢 Blog topic index:", selectedTopic.value);

  // Generate content with selections
  fetch("/generate_blog_with_selection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestData),
  })
    .then((response) => response.json())
    .then((data) => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;

      if (data.success) {
        showToast("success", "Blog content generated successfully!");

        // Hide topic selection
        hideBlogTopicSelection();

        // Switch to blogs tab and refresh content
        const blogsTab = document.getElementById("blogs-tab");
        const blogsTabInstance = new bootstrap.Tab(blogsTab);
        blogsTabInstance.show();

        setTimeout(() => {
          loadContent("blogs");
          refreshStats();
        }, 1000);
      } else {
        showToast("error", data.error || "Failed to generate content");
      }
    })
    .catch((error) => {
      generateBtn.disabled = false;
      generateBtn.innerHTML = originalText;
      showToast("error", "Failed to generate content: " + error.message);
    });
}

// ========================================
// Image Prompt Generation Functions
// ========================================

// Global variable to store current blog data for image prompt generation
let currentBlogData = null;

/**
 * Generate image prompt for blog content
 */
function generateBlogImagePrompt(blogData) {
  console.log("📸 Generating image prompt for blog:", blogData.title);

  // Show loading state
  const button = document.querySelector(".generate-image-prompt-btn");
  if (button) {
    button.disabled = true;
    button.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
  }

  // Call backend API
  fetch("/api/generate_image_prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "blog",
      data: blogData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log("✅ Image prompt generated successfully");
        showImagePromptModal(data.prompt, blogData.title);
      } else {
        console.error("❌ Error:", data.error);
        showToast("error", "Error generating image prompt: " + data.error);
      }
    })
    .catch((error) => {
      console.error("❌ Error generating image prompt:", error);
      showToast("error", "Failed to generate image prompt. Please try again.");
    })
    .finally(() => {
      if (button) {
        button.disabled = false;
        button.innerHTML =
          '<i class="fas fa-image me-1"></i>📸 Generate Image Prompt';
      }
    });
}

/**
 * Show modal with generated image prompt
 */
function showImagePromptModal(prompt, blogTitle) {
  const modalHTML = `
        <div class="modal fade" id="imagePromptModal" tabindex="-1" aria-labelledby="imagePromptModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="imagePromptModalLabel">
                            📸 Generated Image Prompt
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p class="text-muted mb-3">
                            <small>For blog: <strong>${escapeHtml(
                              blogTitle
                            )}</strong></small>
                        </p>
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            Copy this prompt and use it in AI image generators like DALL-E, Midjourney, or Stable Diffusion.
                        </div>
                        <div class="form-group">
                            <label for="imagePromptText" class="form-label fw-bold">Image Prompt:</label>
                            <textarea 
                                class="form-control" 
                                id="imagePromptText" 
                                rows="8" 
                                readonly
                                style="font-family: monospace; font-size: 0.9rem;"
                            >${escapeHtml(prompt)}</textarea>
                        </div>
                        <div class="mt-3">
                            <small class="text-muted">
                                <strong>Tip:</strong> You can edit this prompt before copying if you want to customize it further.
                            </small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="copyImagePromptToClipboard()">
                            <i class="fas fa-clipboard me-2"></i>Copy Prompt
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

  // Remove existing modal if any
  const existingModal = document.getElementById("imagePromptModal");
  if (existingModal) {
    existingModal.remove();
  }

  // Add modal to body
  document.body.insertAdjacentHTML("beforeend", modalHTML);

  // Show modal
  const modal = new bootstrap.Modal(
    document.getElementById("imagePromptModal")
  );
  modal.show();

  // Clean up modal after it's hidden
  document
    .getElementById("imagePromptModal")
    .addEventListener("hidden.bs.modal", function () {
      this.remove();
    });
}

/**
 * Copy image prompt to clipboard
 */
function copyImagePromptToClipboard() {
  const promptText = document.getElementById("imagePromptText");
  if (promptText) {
    promptText.select();
    document.execCommand("copy");

    // Show success notification
    showToast("success", "Image prompt copied to clipboard!");

    // Update button text temporarily
    const copyBtn = event.target.closest("button");
    const originalHTML = copyBtn.innerHTML;
    copyBtn.innerHTML = '<i class="fas fa-check2 me-2"></i>Copied!';
    copyBtn.classList.remove("btn-primary");
    copyBtn.classList.add("btn-success");

    setTimeout(() => {
      copyBtn.innerHTML = originalHTML;
      copyBtn.classList.remove("btn-success");
      copyBtn.classList.add("btn-primary");
    }, 2000);
  }
}

/**
 * Generate image prompt from modal (called when button is clicked in modal)
 */
function generateBlogImagePromptFromModal() {
  if (currentBlogData) {
    generateBlogImagePrompt(currentBlogData);
  } else {
    showToast("error", "No blog data available. Please try again.");
  }
}

/**
 * Show/hide image prompt button based on content type
 */
function showImagePromptButton() {
  const imagePromptBtn = document.querySelector(".generate-image-prompt-btn");
  if (imagePromptBtn) {
    imagePromptBtn.classList.remove("d-none");
  }
}

function hideImagePromptButton() {
  const imagePromptBtn = document.querySelector(".generate-image-prompt-btn");
  if (imagePromptBtn) {
    imagePromptBtn.classList.add("d-none");
  }
  currentBlogData = null;
}

// Add event listener to hide button when modal is closed
document.addEventListener("DOMContentLoaded", function () {
  const contentModal = document.getElementById("contentModal");
  if (contentModal) {
    contentModal.addEventListener("hidden.bs.modal", function () {
      hideImagePromptButton();
    });
  }
});

console.log("✅ Image prompt generation functions loaded");
