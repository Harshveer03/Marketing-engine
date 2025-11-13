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
          // Load LinkedIn Article content by default when social tab is opened
          loadSocialContent("linkedin-article");
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

      // Get quality score and determine badge color
      const qualityScore = blog.quality_score || 0;
      let badgeClass = "bg-secondary";
      if (qualityScore >= 80) {
        badgeClass = "bg-success";
      } else if (qualityScore >= 60) {
        badgeClass = "bg-warning";
      } else if (qualityScore > 0) {
        badgeClass = "bg-danger";
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
                    <div>
                        <button class="btn btn-sm btn-outline-primary" onclick="showBlogModal(${index})">
                            <i class="fas fa-eye me-1"></i>Read Full Post
                        </button>
                        ${
                          qualityScore > 0
                            ? `<span class="badge ${badgeClass} ms-2">Quality: ${Math.round(
                                qualityScore
                              )}%</span>`
                            : ""
                        }
                    </div>
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
      console.log(`📊 Loaded data for platform: ${platform}`, data[platform]);

      if (!data || !data[platform]) {
        contentDiv.innerHTML = `
          <div class="text-center text-muted">
            <i class="${getPlatformIcon(platform)} fa-3x mb-3"></i>
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

      console.log(`📝 Processing ${posts.length} posts for ${platform}`);

      let html = "";

      posts.forEach((post, index) => {
        console.log(`📄 Post ${index}:`, {
          title: post.title,
          hasContent: !!post.content,
          hasCaption: !!post.caption,
          contentLength: post.content ? post.content.length : 0,
          qualityScore: post.quality_score,
        });

        // Determine content field based on platform
        const contentField =
          platform === "linkedin-article" ? post.content : post.caption;
        const platformDisplay = platform
          .replace("-", " ")
          .split(" ")
          .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
          .join(" ");

        // For LinkedIn Article, show preview with "Read Full Article" button
        const isArticle = platform === "linkedin-article";
        const displayContent = isArticle
          ? (contentField || "").substring(0, 300)
          : contentField;

        console.log(
          `✂️ Display content length: ${
            displayContent ? displayContent.length : 0
          }`
        );

        html += `
          <div class="card content-card mb-3">
            <div class="card-header d-flex justify-content-between align-items-center">
              <h6 class="mb-0">
                ${escapeHtml(post.title)}
                ${
                  post.quality_score !== undefined &&
                  post.quality_score !== null &&
                  post.quality_score > 0
                    ? getQualityBadge(post.quality_score)
                    : ""
                }
              </h6>
              <small class="text-muted">
                <i class="${getPlatformIcon(platform)} me-1"></i>
                ${platformDisplay}
                ${post.timestamp ? ` | ${formatDate(post.timestamp)}` : ""}
              </small>
            </div>
            <div class="card-body">
              <p class="card-text">${
                displayContent
                  ? escapeHtml(displayContent)
                  : '<em class="text-muted">No content available</em>'
              }${isArticle && displayContent ? "..." : ""}</p>
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
                ${
                  isArticle
                    ? `
                  <button class="btn btn-sm btn-outline-primary" onclick="showLinkedInArticleModal(${index})">
                    <i class="fas fa-eye me-1"></i>Read Full Article
                  </button>
                `
                    : platform === "linkedin-post"
                    ? `
                  <button class="btn btn-sm btn-outline-primary" onclick="showLinkedInPostModal(${index})">
                    <i class="fas fa-eye me-1"></i>View Full Post
                  </button>
                `
                    : ""
                }
                ${
                  platform !== "linkedin-post"
                    ? `
                <button class="btn btn-sm btn-outline-secondary ${
                  isArticle ? "ms-2" : ""
                }" onclick="copySocialContent('${platform}', ${index})">
                  <i class="fas fa-copy me-1"></i>Copy ${
                    isArticle ? "Article" : "Text"
                  }
                </button>
                `
                    : ""
                }
                ${
                  platform === "twitter"
                    ? `
                <button class="btn btn-sm btn-outline-warning ms-2" id="regenerateTwitterBtn-${index}" onclick="regenerateTwitterPost(${index})">
                  <i class="fas fa-sync-alt me-1"></i>Regenerate
                </button>
                `
                    : ""
                }
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
                ${
                  platform === "linkedin-article" ||
                  platform === "linkedin-post" ||
                  platform === "twitter"
                    ? `
                  <button class="btn btn-sm btn-outline-primary ms-2" onclick="generateSocialImagePrompt('${platform}', ${index})">
                    <i class="fas fa-image me-1"></i>Generate Image Prompt
                  </button>
                `
                    : ""
                }
                ${
                  platform === "youtube"
                    ? `
                  <button class="btn btn-sm btn-outline-danger ms-2" onclick="generateYouTubeVideoPrompt(${index})">
                    <i class="fas fa-video me-1"></i>Generate Video Prompt
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

      // Store social data globally for image prompt generation
      if (!window.socialContentData) {
        window.socialContentData = {};
      }
      window.socialContentData[platform] = posts;

      // Store LinkedIn articles and posts in contentData for modal access (similar to blogs)
      if (platform === "linkedin-article") {
        if (!window.contentData) {
          window.contentData = {};
        }
        window.contentData["linkedin-article"] = posts;
        console.log(
          `✅ Stored ${posts.length} LinkedIn articles in contentData`,
          window.contentData["linkedin-article"]
        );
      }
      
      // Store LinkedIn posts in contentData for modal access
      if (platform === "linkedin-post") {
        if (!window.contentData) {
          window.contentData = {};
        }
        window.contentData["linkedin-post"] = posts;
        console.log(
          `✅ Stored ${posts.length} LinkedIn posts in contentData`,
          window.contentData["linkedin-post"]
        );
      }
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
                ${
                  trend.description
                    ? `<p class="card-text text-muted">${escapeHtml(
                        trend.description
                      )}</p>`
                    : ""
                }
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

  // Set title with quality score badge FIRST
  const qualityScore = blog.quality_score || 0;
  let badgeClass = "bg-secondary";
  let badgeHTML = "";
  if (qualityScore >= 80) {
    badgeClass = "bg-success";
  } else if (qualityScore >= 60) {
    badgeClass = "bg-warning";
  } else if (qualityScore > 0) {
    badgeClass = "bg-danger";
  }

  if (qualityScore > 0) {
    badgeHTML = ` <span class="badge ${badgeClass}">Quality: ${Math.round(
      qualityScore
    )}%</span>`;
  }

  // Update modal content
  document.getElementById("contentModalTitle").innerHTML =
    escapeHtml(blog.title) + badgeHTML;
  document.getElementById("contentModalBody").innerHTML = blogContent.replace(
    /\n/g,
    "<br>"
  );

  // Show the image prompt button for blogs
  showImagePromptButton();

  // Show modal
  const modal = new bootstrap.Modal(document.getElementById("contentModal"));
  modal.show();
}

// LinkedIn Article-specific functions
function showLinkedInArticleModal(index) {
  console.log(`🔍 Opening LinkedIn Article modal for index: ${index}`);
  console.log(`📦 contentData available:`, window.contentData);

  const articles = window.contentData
    ? window.contentData["linkedin-article"]
    : null;

  console.log(`📚 Articles array:`, articles);

  if (!articles || !articles[index]) {
    console.error(
      `❌ Article not found at index ${index}. Available articles:`,
      articles
    );
    showToast(
      "error",
      "LinkedIn Article content not found. Please refresh the page."
    );
    return;
  }

  const article = articles[index];
  console.log(`📄 Article data:`, article);

  let articleContent = article.content || article.caption || "";

  if (!articleContent) {
    console.error(`❌ No content found in article:`, article);
    showToast("error", "Article content is empty.");
    return;
  }

  // Set title with quality score badge
  const qualityScore = article.quality_score || 0;
  let badgeClass = "bg-secondary";
  let badgeHTML = "";
  if (qualityScore >= 80) {
    badgeClass = "bg-success";
  } else if (qualityScore >= 60) {
    badgeClass = "bg-warning";
  } else if (qualityScore > 0) {
    badgeClass = "bg-danger";
  }

  if (qualityScore > 0) {
    badgeHTML = ` <span class="badge ${badgeClass}">Quality: ${Math.round(
      qualityScore
    )}%</span>`;
  }

  // Update modal content
  document.getElementById("contentModalTitle").innerHTML =
    escapeHtml(article.title) + badgeHTML;
  document.getElementById("contentModalBody").innerHTML =
    articleContent.replace(/\n/g, "<br>");

  // Hide the image prompt button for LinkedIn articles (or show if you want)
  hideImagePromptButton();

  // Show modal
  const modal = new bootstrap.Modal(document.getElementById("contentModal"));
  modal.show();

  console.log(`✅ Modal opened successfully`);
}

// LinkedIn Post-specific functions
function showLinkedInPostModal(index) {
  console.log(`🔍 Opening LinkedIn Post modal for index: ${index}`);
  console.log(`📦 contentData available:`, window.contentData);

  const posts = window.contentData
    ? window.contentData["linkedin-post"]
    : null;

  console.log(`📚 Posts array:`, posts);

  if (!posts || !posts[index]) {
    console.error(
      `❌ Post not found at index ${index}. Available posts:`,
      posts
    );
    showToast(
      "error",
      "LinkedIn Post content not found. Please refresh the page."
    );
    return;
  }

  const post = posts[index];
  console.log(`📄 Post data:`, post);

  let postContent = post.caption || post.content || "";

  if (!postContent) {
    console.error(`❌ No content found in post:`, post);
    showToast("error", "Post content is empty.");
    return;
  }

  // Set title with quality score badge
  const qualityScore = post.quality_score || 0;
  let badgeClass = "bg-secondary";
  let badgeHTML = "";
  if (qualityScore >= 80) {
    badgeClass = "bg-success";
  } else if (qualityScore >= 60) {
    badgeClass = "bg-warning";
  } else if (qualityScore > 0) {
    badgeClass = "bg-danger";
  }

  if (qualityScore > 0) {
    badgeHTML = ` <span class="badge ${badgeClass}">Quality: ${Math.round(
      qualityScore
    )}%</span>`;
  }

  // Update modal title
  document.getElementById("linkedinPostModalTitle").innerHTML =
    `<i class="fab fa-linkedin me-2"></i>${escapeHtml(post.title)}` + badgeHTML;

  // Format the post content with LinkedIn-style formatting
  const formattedContent = formatLinkedInPost(postContent);
  document.getElementById("linkedinPostContent").innerHTML = formattedContent;

  // Display hashtags
  const hashtagsContainer = document.getElementById("linkedinPostHashtags");
  if (post.hashtags && post.hashtags.length > 0) {
    const hashtagsHTML = post.hashtags
      .map(tag => {
        // Ensure hashtag starts with #
        const displayTag = tag.startsWith('#') ? tag : `#${tag}`;
        return `<span class="hashtag-badge">${escapeHtml(displayTag)}</span>`;
      })
      .join('');
    hashtagsContainer.innerHTML = hashtagsHTML;
    hashtagsContainer.style.display = 'block';
  } else {
    hashtagsContainer.innerHTML = '';
    hashtagsContainer.style.display = 'none';
  }

  // Store the current post index and original content for copying
  window.currentLinkedInPostIndex = index;
  window.currentLinkedInPostContent = postContent;

  // Show modal
  const modal = new bootstrap.Modal(document.getElementById("linkedinPostModal"));
  modal.show();

  console.log(`✅ LinkedIn Post modal opened successfully`);
}

function formatLinkedInPost(content) {
  if (!content) return "";
  
  // Escape HTML first
  let formatted = escapeHtml(content);
  
  // Convert hashtags to styled spans
  formatted = formatted.replace(/#(\w+)/g, '<span class="hashtag">#$1</span>');
  
  // Convert line breaks to <br> tags
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Wrap in paragraphs for better spacing
  const paragraphs = formatted.split('<br><br>');
  formatted = paragraphs.map(p => p.trim() ? `<p>${p}</p>` : '').join('');
  
  return formatted;
}

function copyLinkedInPostContent() {
  // Get the original content with preserved formatting (line breaks)
  let contentToCopy = window.currentLinkedInPostContent || '';
  
  // Get the current post to access hashtags
  const postIndex = window.currentLinkedInPostIndex;
  const posts = window.contentData ? window.contentData["linkedin-post"] : null;
  
  if (posts && posts[postIndex] && posts[postIndex].hashtags && posts[postIndex].hashtags.length > 0) {
    // Add hashtags to the content
    const hashtags = posts[postIndex].hashtags.map(tag => {
      // Ensure hashtag starts with #
      return tag.startsWith('#') ? tag : `#${tag}`;
    }).join(' ');
    
    // Combine content with hashtags (add two line breaks for spacing)
    contentToCopy = contentToCopy + '\n\n' + hashtags;
  }
  
  if (contentToCopy) {
    copyToClipboard(contentToCopy);
  } else {
    // Fallback to textContent if original content is not available
    const content = document.getElementById("linkedinPostContent").textContent;
    copyToClipboard(content);
  }
}

function copySocialContent(platform, index) {
  // Get the posts for the platform
  const posts = window.socialContentData ? window.socialContentData[platform] : null;
  
  if (!posts || !posts[index]) {
    showToast('error', 'Unable to copy: Post data not found');
    return;
  }
  
  const post = posts[index];
  let contentToCopy = post.caption || post.content || '';
  
  // Add hashtags if available
  if (post.hashtags && post.hashtags.length > 0) {
    const hashtags = post.hashtags.map(tag => {
      return tag.startsWith('#') ? tag : `#${tag}`;
    }).join(' ');
    
    contentToCopy = contentToCopy + '\n\n' + hashtags;
  }
  
  if (contentToCopy) {
    copyToClipboard(contentToCopy);
  } else {
    showToast('error', 'No content to copy');
  }
}

function regenerateLinkedInPost() {
  console.log('🔄 Regenerate button clicked');
  
  const postIndex = window.currentLinkedInPostIndex;
  
  if (postIndex === undefined || postIndex === null) {
    showToast('error', 'Unable to regenerate: Post index not found');
    return;
  }
  
  const posts = window.contentData ? window.contentData["linkedin-post"] : null;
  
  if (!posts || !posts[postIndex]) {
    showToast('error', 'Unable to regenerate: Post data not found');
    return;
  }
  
  const post = posts[postIndex];
  
  // Disable the regenerate button and show loading state
  const regenerateBtn = document.getElementById('regenerateLinkedInPostBtn');
  const originalBtnText = regenerateBtn.innerHTML;
  regenerateBtn.disabled = true;
  regenerateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Regenerating...';
  
  // Show loading state in modal content
  const contentDiv = document.getElementById('linkedinPostContent');
  const originalContent = contentDiv.innerHTML;
  contentDiv.innerHTML = `
    <div class="text-center py-5">
      <div class="spinner-border text-primary mb-3" role="status">
        <span class="visually-hidden">Regenerating...</span>
      </div>
      <p class="text-muted">Regenerating post content...</p>
    </div>
  `;
  
  console.log('📤 Sending regenerate request for post:', {
    post_index: postIndex,
    title: post.title,
    industry: post.industry,
    tone: post.tone,
    audience: post.audience
  });
  
  // Call backend to regenerate
  fetch('/regenerate_linkedin_post', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      post_index: postIndex,
      title: post.title,
      industry: post.industry || 'IT & Dev',
      tone: post.tone || 'professional',
      audience: post.audience || 'Founders'
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      console.log('✅ Post regenerated successfully:', data.post);
      
      // Update the stored data
      posts[postIndex] = data.post;
      window.contentData["linkedin-post"] = posts;
      
      // Update the modal content
      const newContent = data.post.caption || data.post.content || '';
      const formattedContent = formatLinkedInPost(newContent);
      contentDiv.innerHTML = formattedContent;
      
      // Update the stored content for copying
      window.currentLinkedInPostContent = newContent;
      
      // Update hashtags
      const hashtagsContainer = document.getElementById("linkedinPostHashtags");
      if (data.post.hashtags && data.post.hashtags.length > 0) {
        const hashtagsHTML = data.post.hashtags
          .map(tag => {
            const displayTag = tag.startsWith('#') ? tag : `#${tag}`;
            return `<span class="hashtag-badge">${escapeHtml(displayTag)}</span>`;
          })
          .join('');
        hashtagsContainer.innerHTML = hashtagsHTML;
        hashtagsContainer.style.display = 'block';
      } else {
        hashtagsContainer.innerHTML = '';
        hashtagsContainer.style.display = 'none';
      }
      
      // Update quality score in title if present
      const qualityScore = data.post.quality_score || 0;
      let badgeHTML = '';
      if (qualityScore > 0) {
        let badgeClass = 'bg-secondary';
        if (qualityScore >= 80) {
          badgeClass = 'bg-success';
        } else if (qualityScore >= 60) {
          badgeClass = 'bg-warning';
        } else {
          badgeClass = 'bg-danger';
        }
        badgeHTML = ` <span class="badge ${badgeClass}">Quality: ${Math.round(qualityScore)}%</span>`;
      }
      
      document.getElementById('linkedinPostModalTitle').innerHTML =
        `<i class="fab fa-linkedin me-2"></i>${escapeHtml(data.post.title)}` + badgeHTML;
      
      showToast('success', 'LinkedIn Post regenerated successfully!');
      
      // Reload the LinkedIn post content in the background to update the card view
      setTimeout(() => {
        loadSocialContent('linkedin-post');
      }, 1000);
    } else {
      contentDiv.innerHTML = originalContent;
      showToast('error', data.error || 'Failed to regenerate post');
    }
    
    // Re-enable the button
    regenerateBtn.disabled = false;
    regenerateBtn.innerHTML = originalBtnText;
  })
  .catch(error => {
    console.error('❌ Regenerate error:', error);
    contentDiv.innerHTML = originalContent;
    regenerateBtn.disabled = false;
    regenerateBtn.innerHTML = originalBtnText;
    showToast('error', 'Failed to regenerate post: ' + error.message);
  });
}

function regenerateTwitterPost(index) {
  console.log('🔄 Regenerate Twitter post button clicked for index:', index);
  
  const posts = window.socialContentData ? window.socialContentData['twitter'] : null;
  
  if (!posts || !posts[index]) {
    showToast('error', 'Unable to regenerate: Twitter post data not found');
    return;
  }
  
  const post = posts[index];
  
  // Get the button and show loading state
  const regenerateBtn = document.getElementById(`regenerateTwitterBtn-${index}`);
  const originalBtnText = regenerateBtn.innerHTML;
  regenerateBtn.disabled = true;
  regenerateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Regenerating...';
  
  console.log('📤 Sending regenerate request for Twitter post:', {
    post_index: index,
    title: post.title,
    industry: post.industry,
    tone: post.tone,
    audience: post.audience
  });
  
  // Show loading toast
  showToast('info', 'Regenerating Twitter post... This may take a few moments.');
  
  // Call backend to regenerate
  fetch('/regenerate_twitter_post', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      post_index: index,
      title: post.title,
      industry: post.industry || 'IT & Dev',
      tone: post.tone || 'professional',
      audience: post.audience || 'Founders'
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      console.log('✅ Twitter post regenerated successfully:', data.post);
      
      // Update the stored data
      posts[index] = data.post;
      window.socialContentData['twitter'] = posts;
      
      showToast('success', 'Twitter post regenerated successfully!');
      
      // Reload the Twitter content to update the card view
      setTimeout(() => {
        loadSocialContent('twitter');
      }, 1000);
    } else {
      // Re-enable button on error
      regenerateBtn.disabled = false;
      regenerateBtn.innerHTML = originalBtnText;
      showToast('error', data.error || 'Failed to regenerate Twitter post');
    }
  })
  .catch(error => {
    console.error('❌ Regenerate Twitter error:', error);
    // Re-enable button on error
    regenerateBtn.disabled = false;
    regenerateBtn.innerHTML = originalBtnText;
    showToast('error', 'Failed to regenerate Twitter post: ' + error.message);
  });
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
    "linkedin-article": "fab fa-linkedin",
    "linkedin-post": "fab fa-linkedin",
    twitter: "fab fa-x-twitter",
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

function getQualityBadge(qualityScore) {
  if (!qualityScore || qualityScore === 0) return "";

  let badgeClass = "bg-secondary";
  if (qualityScore >= 80) {
    badgeClass = "bg-success";
  } else if (qualityScore >= 60) {
    badgeClass = "bg-warning";
  } else {
    badgeClass = "bg-danger";
  }

  return `<span class="badge ${badgeClass} ms-2">Quality: ${Math.round(
    qualityScore
  )}%</span>`;
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

// Loading Modal Functions
function showLoadingModal(message) {
  // Reset to loading state
  document.getElementById("loading-state").classList.remove("d-none");
  document.getElementById("loading-error-state").classList.add("d-none");
  
  // Set message
  document.getElementById("loading-message").textContent = message || "Processing...";
  
  // Show modal
  const loadingModal = new bootstrap.Modal(document.getElementById("loadingModal"));
  loadingModal.show();
}

function hideLoadingModal() {
  const loadingModal = bootstrap.Modal.getInstance(document.getElementById("loadingModal"));
  if (loadingModal) {
    loadingModal.hide();
  }
}

function showLoadingError(errorMessage) {
  // Hide loading state
  document.getElementById("loading-state").classList.add("d-none");
  
  // Show error state
  document.getElementById("loading-error-state").classList.remove("d-none");
  document.getElementById("loading-error-message").textContent = errorMessage;
}

// Social Mode Selection Functions
function showSocialModeSelection() {
  console.log("showSocialModeSelection called - showing mode selection modal");
  const modal = new bootstrap.Modal(document.getElementById("socialModeModal"));
  modal.show();
}

function selectSocialMode(mode) {
  console.log(`🎯 DEBUG: User selected social mode: ${mode}`);

  // Hide mode selection modal
  const modeModal = bootstrap.Modal.getInstance(
    document.getElementById("socialModeModal")
  );
  if (modeModal) {
    modeModal.hide();
  }

  if (mode === "automatic") {
    // NEW: Show industry and platform selection modal
    setTimeout(() => {
      showSocialIndustryPlatformSelection();
    }, 300);
  } else if (mode === "manual") {
    // Keep manual flow as-is
    setTimeout(() => {
      showManualSocialTopicInput();
    }, 300);
  }
}

function showManualSocialTopicInput() {
  console.log("showManualSocialTopicInput called - showing manual input modal");

  // Clear previous input
  document.getElementById("manualSocialTopicInput").value = "";

  // Set default industry
  setDefaultIndustry();

  // Show modal
  const modal = new bootstrap.Modal(
    document.getElementById("manualSocialTopicModal")
  );
  modal.show();
}

function generateSocialManual() {
  const topic = document.getElementById("manualSocialTopicInput").value.trim();
  const industry = document.getElementById("manualSocialIndustrySelect").value;
  const tone = document.getElementById("manualSocialToneSelect").value;
  const audience = document.getElementById("manualSocialAudienceSelect").value;

  // Get selected platform from dropdown
  const selectedPlatform = document.getElementById(
    "manualPlatformSelect"
  ).value;

  if (!topic) {
    showToast("error", "Please enter a topic");
    return;
  }

  console.log(`📝 Manual social generation:`, {
    topic,
    industry,
    tone,
    audience,
    platform: selectedPlatform,
  });

  // Hide modal
  const modal = bootstrap.Modal.getInstance(
    document.getElementById("manualSocialTopicModal")
  );
  if (modal) {
    modal.hide();
  }

  // Show loading toast
  const platformName = selectedPlatform
    .replace("-", " ")
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
  showToast(
    "info",
    `Fetching trends for "${topic}" and generating ${platformName} content... This may take a few moments.`
  );

  // Call backend API
  fetch("/generate_social_manual", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      topic: topic,
      industry: industry,
      tone: tone,
      audience: audience,
      platform: selectedPlatform,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showToast("success", `${platformName} content generated successfully!`);

        // Switch to social tab and refresh content
        const socialTab = document.getElementById("social-tab");
        const socialTabInstance = new bootstrap.Tab(socialTab);
        socialTabInstance.show();

        setTimeout(() => {
          // Load the generated platform content
          loadSocialContent(selectedPlatform);
          refreshStats();
        }, 1000);
      } else {
        showToast("error", data.error || "Failed to generate social content");
      }
    })
    .catch((error) => {
      showToast("error", "Failed to generate social content: " + error.message);
    });
}

// Dashboard Topic Selection Functions (Automatic Mode)
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
  const tone = document.getElementById("toneSelect").value;
  const audience = document.getElementById("audienceSelect").value;

  // Use stored industry and platform from previous steps
  const industry = selectedSocialIndustry || "IT & Dev";
  const selectedPlatform = selectedSocialPlatform || "linkedin-article";
  
  console.log(`📊 DEBUG: Using stored industry: ${industry}, platform: ${selectedPlatform}`);

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

  const platformName = selectedPlatform
    .replace("-", " ")
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
  showToast(
    "info",
    `Generating ${platformName} content... This may take a few moments.`
  );

  // Debug: Log what we're sending
  const requestData = {
    topic_index: parseInt(selectedTopic.value),
    industry: industry,
    tone: tone,
    audience: audience,
    platform: selectedPlatform,
  };

  console.log("🚀 Sending request:", requestData);
  console.log("📝 Selected platform:", selectedPlatform);

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
        showToast("success", `${platformName} content generated successfully!`);

        // Hide topic selection
        hideTopicSelection();

        // Switch to social tab and refresh content
        const socialTab = document.getElementById("social-tab");
        const socialTabInstance = new bootstrap.Tab(socialTab);
        socialTabInstance.show();

        setTimeout(() => {
          // Load the generated platform content
          loadSocialContent(selectedPlatform);
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

// Blog Mode Selection Functions
function showBlogModeSelection() {
  console.log("showBlogModeSelection called - showing mode selection modal");
  const modal = new bootstrap.Modal(document.getElementById("blogModeModal"));
  modal.show();
}

function selectBlogMode(mode) {
  console.log(`User selected blog mode: ${mode}`);

  // Hide mode selection modal
  const modeModal = bootstrap.Modal.getInstance(
    document.getElementById("blogModeModal")
  );
  if (modeModal) {
    modeModal.hide();
  }

  if (mode === "automatic") {
    // Existing automatic flow
    setTimeout(() => {
      showBlogTopicSelection();
    }, 300);
  } else if (mode === "manual") {
    // New manual flow
    setTimeout(() => {
      showManualBlogTopicInput();
    }, 300);
  }
}

function showManualBlogTopicInput() {
  console.log("showManualBlogTopicInput called - showing manual input modal");

  // Clear previous input
  document.getElementById("manualBlogTopicInput").value = "";

  // Set default industry
  setDefaultIndustry();

  // Show modal
  const modal = new bootstrap.Modal(
    document.getElementById("manualBlogTopicModal")
  );
  modal.show();
}

function generateBlogManual() {
  const topic = document.getElementById("manualBlogTopicInput").value.trim();
  const industry = document.getElementById("manualBlogIndustrySelect").value;
  const tone = document.getElementById("manualBlogToneSelect").value;
  const audience = document.getElementById("manualBlogAudienceSelect").value;

  if (!topic) {
    showToast("error", "Please enter a topic");
    return;
  }

  console.log(`📝 Manual blog generation:`, {
    topic,
    industry,
    tone,
    audience,
  });

  // Hide modal
  const modal = bootstrap.Modal.getInstance(
    document.getElementById("manualBlogTopicModal")
  );
  if (modal) {
    modal.hide();
  }

  // Show loading toast
  showToast(
    "info",
    `Fetching trends for "${topic}"... This may take a few moments.`
  );

  // Call backend API
  fetch("/generate_blog_manual", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      topic: topic,
      industry: industry,
      tone: tone,
      audience: audience,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showToast("success", `Blog "${data.title}" generated successfully!`);

        // Switch to blogs tab and refresh content
        const blogsTab = document.getElementById("blogs-tab");
        const blogsTabInstance = new bootstrap.Tab(blogsTab);
        blogsTabInstance.show();

        setTimeout(() => {
          loadContent("blogs");
          refreshStats();
        }, 1000);
      } else {
        showToast("error", data.error || "Failed to generate blog");
      }
    })
    .catch((error) => {
      showToast("error", "Failed to generate blog: " + error.message);
    });
}

// Blog Topic Selection Functions (Automatic Mode)
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
  const tone = document.getElementById("blogToneSelect").value;
  const audience = document.getElementById("blogAudienceSelect").value;
  
  // Use the stored industry from previous step
  const industry = selectedBlogIndustry || "IT & Dev";
  console.log(`📊 DEBUG: Using stored industry: ${industry}`);

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

// ========================================
// Social Media Image Prompt Generation Functions
// ========================================

/**
 * Generate image prompt for LinkedIn or X (Twitter) content
 */
function generateSocialImagePrompt(platform, postIndex) {
  console.log(
    `📸 Generating ${platform} image prompt for post index:`,
    postIndex
  );

  // Get the post data
  if (!window.socialContentData || !window.socialContentData[platform]) {
    showToast("error", "Post data not found. Please refresh the page.");
    return;
  }

  const posts = window.socialContentData[platform];
  if (!posts[postIndex]) {
    showToast("error", "Post not found. Please try again.");
    return;
  }

  const postData = posts[postIndex];

  // Show loading toast
  showToast("info", `Generating ${platform} image prompt...`);

  // Call backend API
  fetch("/api/generate_image_prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: platform,
      data: postData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log(`✅ ${platform} image prompt generated successfully`);
        showImagePromptModal(data.prompt, postData.title, platform);
      } else {
        console.error("❌ Error:", data.error);
        showToast("error", "Error generating image prompt: " + data.error);
      }
    })
    .catch((error) => {
      console.error(`❌ Error generating ${platform} image prompt:`, error);
      showToast("error", "Failed to generate image prompt. Please try again.");
    });
}

/**
 * Generate LinkedIn image prompt (convenience function)
 */
function generateLinkedInImagePrompt(linkedinData) {
  console.log("📸 Generating LinkedIn image prompt:", linkedinData.title);

  // Call backend API
  fetch("/api/generate_image_prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "linkedin",
      data: linkedinData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log("✅ LinkedIn image prompt generated successfully");
        showImagePromptModal(data.prompt, linkedinData.title, "linkedin");
      } else {
        console.error("❌ Error:", data.error);
        showToast("error", "Error generating image prompt: " + data.error);
      }
    })
    .catch((error) => {
      console.error("❌ Error generating LinkedIn image prompt:", error);
      showToast("error", "Failed to generate image prompt. Please try again.");
    });
}

/**
 * Generate X (Twitter) image prompt (convenience function)
 */
function generateTwitterImagePrompt(twitterData) {
  console.log("📸 Generating X (Twitter) image prompt:", twitterData.title);

  // Call backend API
  fetch("/api/generate_image_prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "twitter",
      data: twitterData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log("✅ X (Twitter) image prompt generated successfully");
        showImagePromptModal(data.prompt, twitterData.title, "twitter");
      } else {
        console.error("❌ Error:", data.error);
        showToast("error", "Error generating image prompt: " + data.error);
      }
    })
    .catch((error) => {
      console.error("❌ Error generating X (Twitter) image prompt:", error);
      showToast("error", "Failed to generate image prompt. Please try again.");
    });
}

console.log("✅ Social media image prompt generation functions loaded");

// ========================================
// YouTube Video Prompt Generation Functions
// ========================================

/**
 * Generate video prompt for YouTube content
 */
function generateYouTubeVideoPrompt(postIndex) {
  console.log(`🎬 Generating YouTube video prompt for post index:`, postIndex);

  // Get the post data
  if (!window.socialContentData || !window.socialContentData["youtube"]) {
    showToast("error", "YouTube post data not found. Please refresh the page.");
    return;
  }

  const posts = window.socialContentData["youtube"];
  if (!posts[postIndex]) {
    showToast("error", "YouTube post not found. Please try again.");
    return;
  }

  const postData = posts[postIndex];

  // Show loading toast
  showToast("info", "Generating YouTube video prompt...");

  // Call backend API
  fetch("/api/generate_image_prompt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "youtube",
      data: postData,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log("✅ YouTube video prompt generated successfully");
        showVideoPromptModal(data.prompt, postData.title);
      } else {
        console.error("❌ Error:", data.error);
        showToast("error", "Error generating video prompt: " + data.error);
      }
    })
    .catch((error) => {
      console.error("❌ Error generating YouTube video prompt:", error);
      showToast("error", "Failed to generate video prompt. Please try again.");
    });
}

/**
 * Show modal with generated video prompt
 */
function showVideoPromptModal(prompt, videoTitle) {
  const modalHTML = `
        <div class="modal fade" id="videoPromptModal" tabindex="-1" aria-labelledby="videoPromptModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="videoPromptModalLabel">
                            🎬 Generated Video Prompt
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p class="text-muted mb-3">
                            <small>For YouTube video: <strong>${escapeHtml(
                              videoTitle
                            )}</strong></small>
                        </p>
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            Copy this prompt and use it in AI video generators like Runway, Pika, Synthesia, or similar tools.
                        </div>
                        <div class="form-group">
                            <label for="videoPromptText" class="form-label fw-bold">Video Prompt:</label>
                            <textarea 
                                class="form-control" 
                                id="videoPromptText" 
                                rows="12" 
                                readonly
                                style="font-family: monospace; font-size: 0.9rem;"
                            >${escapeHtml(prompt)}</textarea>
                        </div>
                        <div class="mt-3">
                            <small class="text-muted">
                                <strong>Tip:</strong> This prompt is optimized for AI video generators. You can edit it before copying if you want to customize it further.
                            </small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-danger" onclick="copyVideoPromptToClipboard()">
                            <i class="fas fa-clipboard me-2"></i>Copy Video Prompt
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

  // Remove existing modal if any
  const existingModal = document.getElementById("videoPromptModal");
  if (existingModal) {
    existingModal.remove();
  }

  // Add modal to body
  document.body.insertAdjacentHTML("beforeend", modalHTML);

  // Show modal
  const modal = new bootstrap.Modal(
    document.getElementById("videoPromptModal")
  );
  modal.show();

  // Clean up modal after it's hidden
  document
    .getElementById("videoPromptModal")
    .addEventListener("hidden.bs.modal", function () {
      this.remove();
    });
}

/**
 * Copy video prompt to clipboard
 */
function copyVideoPromptToClipboard() {
  const promptText = document.getElementById("videoPromptText");
  if (promptText) {
    promptText.select();
    document.execCommand("copy");

    // Show success notification
    showToast("success", "Video prompt copied to clipboard!");

    // Update button text temporarily
    const copyBtn = event.target.closest("button");
    const originalHTML = copyBtn.innerHTML;
    copyBtn.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
    copyBtn.classList.remove("btn-danger");
    copyBtn.classList.add("btn-success");

    setTimeout(() => {
      copyBtn.innerHTML = originalHTML;
      copyBtn.classList.remove("btn-success");
      copyBtn.classList.add("btn-danger");
    }, 2000);
  }
}

console.log("✅ YouTube video prompt generation functions loaded");


// ========================================
// Blog Trend Slider Functions
// ========================================

// Global variables to store fetched trends and selected industry
let fetchedBlogTrends = [];
let selectedBlogIndustry = "";

/**
 * Show blog mode selection modal
 */
function showBlogModeSelection() {
  console.log("🎯 DEBUG: showBlogModeSelection called");
  const modal = new bootstrap.Modal(document.getElementById("blogModeModal"));
  modal.show();
}

/**
 * Handle blog mode selection
 */
function selectBlogMode(mode) {
  console.log(`🎯 DEBUG: User selected blog mode: ${mode}`);

  // Hide mode selection modal
  const modeModal = bootstrap.Modal.getInstance(
    document.getElementById("blogModeModal")
  );
  if (modeModal) {
    modeModal.hide();
  }

  if (mode === "automatic") {
    // Show industry selection modal
    setTimeout(() => {
      showBlogIndustrySelection();
    }, 300);
  } else if (mode === "manual") {
    // Existing manual flow
    setTimeout(() => {
      showManualBlogTopicInput();
    }, 300);
  }
}

/**
 * Show blog industry selection modal
 */
function showBlogIndustrySelection() {
  console.log("📋 DEBUG: Showing blog industry selection modal");
  const modal = new bootstrap.Modal(
    document.getElementById("blogIndustryModal")
  );
  modal.show();
}

/**
 * Fetch blog trends for selected industry
 */
function fetchBlogTrends() {
  const industry = document.getElementById("blogTargetIndustrySelect").value;
  console.log(`🔍 DEBUG: Fetching trends for industry: ${industry}`);

  // Store selected industry globally
  selectedBlogIndustry = industry;
  console.log(`💾 DEBUG: Stored selected industry: ${selectedBlogIndustry}`);

  // Hide industry modal
  const industryModal = bootstrap.Modal.getInstance(
    document.getElementById("blogIndustryModal")
  );
  if (industryModal) {
    industryModal.hide();
  }

  // Show loading modal with custom message
  showLoadingModal("Analyzing industry trends...");

  // Call backend to fetch trends
  fetch("/fetch_blog_trends", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      industry: industry,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log(`✅ DEBUG: Fetched ${data.trends.length} trends`);
        fetchedBlogTrends = data.trends;

        // Hide loading modal
        hideLoadingModal();

        // Show trend slider modal
        setTimeout(() => {
          showBlogTrendSlider();
        }, 300);
      } else {
        console.error(`❌ DEBUG: Error fetching trends: ${data.error}`);
        // Show error in loading modal
        showLoadingError(
          `Failed to fetch trends: ${data.error || "Unknown error"}. Please try again.`
        );
      }
    })
    .catch((error) => {
      console.error(`❌ DEBUG: Network error fetching trends:`, error);
      // Show error in loading modal
      showLoadingError(
        `Network error while fetching trends: ${error.message}. Please check your connection and try again.`
      );
    });
}

/**
 * Show blog trend slider modal
 */
function showBlogTrendSlider() {
  console.log("🎚️ DEBUG: Showing trend slider modal");

  // Reset slider to default value (0)
  const slider = document.getElementById("trendInfluenceSlider");
  slider.value = 0;
  document.getElementById("sliderValueDisplay").textContent = "Value: 0";

  // Add event listener to update display
  slider.addEventListener("input", function () {
    document.getElementById("sliderValueDisplay").textContent =
      `Value: ${this.value}`;
    console.log(`🎚️ DEBUG: Slider value changed to: ${this.value}`);
  });

  const modal = new bootstrap.Modal(
    document.getElementById("blogTrendSliderModal")
  );
  modal.show();
}

/**
 * Generate blog topics with slider value
 */
function generateBlogTopicsWithSlider() {
  const sliderValue = parseInt(
    document.getElementById("trendInfluenceSlider").value
  );
  const industry = document.getElementById("blogTargetIndustrySelect").value;

  console.log(`📝 DEBUG: Generating topics with slider value: ${sliderValue}`);
  console.log(`📊 DEBUG: Industry: ${industry}`);
  console.log(`📊 DEBUG: Trends count: ${fetchedBlogTrends.length}`);

  // Hide slider modal
  const sliderModal = bootstrap.Modal.getInstance(
    document.getElementById("blogTrendSliderModal")
  );
  if (sliderModal) {
    sliderModal.hide();
  }

  // Show topic selection section with loading
  const blogTopicSection = document.getElementById("blogTopicSelectionSection");
  blogTopicSection.classList.remove("d-none");
  document.getElementById("blogTopicLoading").classList.remove("d-none");
  document.getElementById("blogTopicSelectionContent").classList.add("d-none");

  // Show loading toast
  showToast("info", "Generating topics...");

  // Call backend to generate topics
  fetch("/generate_blog_topics", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      trend_influence: sliderValue,
      trends_data: fetchedBlogTrends,
      industry: industry,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.topics) {
        console.log(`✅ DEBUG: Generated ${data.topics.length} topics`);
        console.log("📋 DEBUG: Topics:", data.topics);

        // Display topics with badges
        displayBlogTopicsWithBadges(data.topics);

        // Show success toast
        showToast("success", "Topics generated successfully!");
      } else {
        console.error(`❌ DEBUG: Error generating topics: ${data.error}`);
        showToast("error", data.error || "Failed to generate topics");
        hideBlogTopicSelection();
      }
    })
    .catch((error) => {
      console.error(`❌ DEBUG: Network error generating topics:`, error);
      showToast("error", "Failed to generate topics: " + error.message);
      hideBlogTopicSelection();
    });
}

/**
 * Display blog topics with trend follower/setter badges
 */
function displayBlogTopicsWithBadges(topics) {
  console.log("🏷️ DEBUG: Displaying topics with badges");

  const topicList = document.getElementById("blogTopicList");
  topicList.innerHTML = "";

  topics.forEach((topic, index) => {
    const relatedCount = topic.related_news ? topic.related_news.length : 0;
    const relevanceScore = topic.relevance_score || 0;
    const topicType = topic.type || "trend_follower";

    console.log(`📝 DEBUG: Topic ${index}: ${topic.title} | Type: ${topicType}`);

    // Determine badge based on type
    let badgeHTML = "";
    if (topicType === "trend_follower") {
      badgeHTML = `<span class="badge bg-primary me-2" style="font-size: 0.75rem;">
        <i class="fas fa-chart-line me-1"></i>Trend Follower
      </span>`;
    } else if (topicType === "trend_setter") {
      badgeHTML = `<span class="badge bg-success me-2" style="font-size: 0.75rem;">
        <i class="fas fa-lightbulb me-1"></i>Trend Setter
      </span>`;
    }

    topicList.innerHTML += `
      <div class="card mb-3 blog-topic-card" style="cursor: pointer;" onclick="selectBlogTopic(${index})">
        <div class="card-body">
          <div class="form-check">
            <input class="form-check-input" type="radio" name="dashboardBlogTopic" value="${index}" id="dashboardBlogTopic${index}">
            <label class="form-check-label w-100" for="dashboardBlogTopic${index}">
              <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                  <div class="mb-2">
                    ${badgeHTML}
                  </div>
                  <h6 class="mb-1">${escapeHtml(topic.title)}</h6>
                  <small class="text-muted">
                    <i class="fas fa-newspaper me-1"></i>${relatedCount} related articles
                  </small>
                </div>
                <span class="badge bg-info ms-2">
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
  document.getElementById("blogTopicSelectionContent").classList.remove("d-none");
}

console.log("✅ Blog trend slider functions loaded");


// ========================================
// Social Media Trend Slider Functions
// ========================================

// Global variables to store fetched trends, selected industry and platform
let fetchedSocialTrends = [];
let selectedSocialIndustry = "";
let selectedSocialPlatform = "";

/**
 * Show social industry and platform selection modal
 */
function showSocialIndustryPlatformSelection() {
  console.log("📋 DEBUG: Showing social industry and platform selection modal");
  const modal = new bootstrap.Modal(
    document.getElementById("socialIndustryPlatformModal")
  );
  modal.show();
}

/**
 * Fetch social trends for selected industry
 */
function fetchSocialTrends() {
  const industry = document.getElementById("socialTargetIndustrySelect").value;
  const platform = document.getElementById("socialTargetPlatformSelect").value;
  console.log(`🔍 DEBUG: Fetching trends for industry: ${industry}, platform: ${platform}`);

  // Store selected industry and platform globally
  selectedSocialIndustry = industry;
  selectedSocialPlatform = platform;
  console.log(`💾 DEBUG: Stored selected industry: ${selectedSocialIndustry}, platform: ${selectedSocialPlatform}`);

  // Hide industry/platform modal
  const industryModal = bootstrap.Modal.getInstance(
    document.getElementById("socialIndustryPlatformModal")
  );
  if (industryModal) {
    industryModal.hide();
  }

  // Show loading modal with custom message
  showLoadingModal("Analyzing industry trends...");

  // Call backend to fetch trends
  fetch("/fetch_social_trends", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      industry: industry,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        console.log(`✅ DEBUG: Fetched ${data.trends.length} trends`);
        fetchedSocialTrends = data.trends;

        // Hide loading modal
        hideLoadingModal();

        // Show trend slider modal
        setTimeout(() => {
          showSocialTrendSlider();
        }, 300);
      } else {
        console.error(`❌ DEBUG: Error fetching trends: ${data.error}`);
        // Show error in loading modal
        showLoadingError(
          `Failed to fetch trends: ${data.error || "Unknown error"}. Please try again.`
        );
      }
    })
    .catch((error) => {
      console.error(`❌ DEBUG: Network error fetching trends:`, error);
      // Show error in loading modal
      showLoadingError(
        `Network error while fetching trends: ${error.message}. Please check your connection and try again.`
      );
    });
}

/**
 * Show social trend slider modal
 */
function showSocialTrendSlider() {
  console.log("🎚️ DEBUG: Showing social trend slider modal");

  // Reset slider to default value (0)
  const slider = document.getElementById("socialTrendInfluenceSlider");
  slider.value = 0;
  document.getElementById("socialSliderValueDisplay").textContent = "Value: 0";

  // Add event listener to update display
  slider.addEventListener("input", function () {
    document.getElementById("socialSliderValueDisplay").textContent =
      `Value: ${this.value}`;
    console.log(`🎚️ DEBUG: Social slider value changed to: ${this.value}`);
  });

  const modal = new bootstrap.Modal(
    document.getElementById("socialTrendSliderModal")
  );
  modal.show();
}

/**
 * Generate social topics with slider value
 */
function generateSocialTopicsWithSlider() {
  const sliderValue = parseInt(
    document.getElementById("socialTrendInfluenceSlider").value
  );
  const industry = selectedSocialIndustry;
  const platform = selectedSocialPlatform;

  console.log(`📝 DEBUG: Generating social topics with slider value: ${sliderValue}`);
  console.log(`📊 DEBUG: Industry: ${industry}, Platform: ${platform}`);
  console.log(`📊 DEBUG: Trends count: ${fetchedSocialTrends.length}`);

  // Hide slider modal
  const sliderModal = bootstrap.Modal.getInstance(
    document.getElementById("socialTrendSliderModal")
  );
  if (sliderModal) {
    sliderModal.hide();
  }

  // Show topic selection section with loading
  const topicSection = document.getElementById("topicSelectionSection");
  topicSection.classList.remove("d-none");
  document.getElementById("topicLoading").classList.remove("d-none");
  document.getElementById("topicSelectionContent").classList.add("d-none");

  // Show loading toast
  showToast("info", "Generating topics...");

  // Call backend to generate topics
  fetch("/generate_social_topics", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      trend_influence: sliderValue,
      trends_data: fetchedSocialTrends,
      industry: industry,
      platform: platform,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.topics) {
        console.log(`✅ DEBUG: Generated ${data.topics.length} topics`);
        console.log("📋 DEBUG: Topics:", data.topics);

        // Display topics with badges
        displaySocialTopicsWithBadges(data.topics);

        // Show success toast
        showToast("success", "Topics generated successfully!");
      } else {
        console.error(`❌ DEBUG: Error generating topics: ${data.error}`);
        showToast("error", data.error || "Failed to generate topics");
        hideTopicSelection();
      }
    })
    .catch((error) => {
      console.error(`❌ DEBUG: Network error generating topics:`, error);
      showToast("error", "Failed to generate topics: " + error.message);
      hideTopicSelection();
    });
}

/**
 * Display social topics with trend follower/setter badges
 */
function displaySocialTopicsWithBadges(topics) {
  console.log("🏷️ DEBUG: Displaying social topics with badges");

  const topicList = document.getElementById("topicList");
  topicList.innerHTML = "";

  topics.forEach((topic, index) => {
    const relatedCount = topic.related_news ? topic.related_news.length : 0;
    const relevanceScore = topic.relevance_score || 0;
    const topicType = topic.type || "trend_follower";

    console.log(`📝 DEBUG: Topic ${index}: ${topic.title} | Type: ${topicType} | Platform: ${selectedSocialPlatform}`);

    // Determine badge based on type
    let badgeHTML = "";
    if (topicType === "trend_follower") {
      badgeHTML = `<span class="badge bg-primary me-2" style="font-size: 0.75rem;">
        <i class="fas fa-chart-line me-1"></i>Trend Follower
      </span>`;
    } else if (topicType === "trend_setter") {
      badgeHTML = `<span class="badge bg-success me-2" style="font-size: 0.75rem;">
        <i class="fas fa-lightbulb me-1"></i>Trend Setter
      </span>`;
    }

    // Platform display badge
    const platformDisplay = selectedSocialPlatform
      .replace("-", " ")
      .split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
    
    const platformBadgeHTML = `<span class="badge bg-info me-2" style="font-size: 0.75rem;">
      <i class="${getPlatformIcon(selectedSocialPlatform)} me-1"></i>${platformDisplay}
    </span>`;

    topicList.innerHTML += `
      <div class="card mb-3 topic-card" style="cursor: pointer;" onclick="selectTopic(${index})">
        <div class="card-body">
          <div class="form-check">
            <input class="form-check-input" type="radio" name="dashboardTopic" value="${index}" id="dashboardTopic${index}">
            <label class="form-check-label w-100" for="dashboardTopic${index}">
              <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                  <div class="mb-2">
                    ${badgeHTML}
                    ${platformBadgeHTML}
                  </div>
                  <h6 class="mb-1">${escapeHtml(topic.title)}</h6>
                  <small class="text-muted">
                    <i class="fas fa-newspaper me-1"></i>${relatedCount} related articles
                  </small>
                </div>
                <span class="badge bg-info ms-2">
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
  document.getElementById("topicLoading").classList.add("d-none");
  document.getElementById("topicSelectionContent").classList.remove("d-none");
}

console.log("✅ Social media trend slider functions loaded");
