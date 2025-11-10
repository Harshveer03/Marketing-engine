# 🔧 File Naming Fix - Social Content

## Issue Identified

The manual social generation was creating files with **hyphens** instead of **underscores**, causing duplicate files and inconsistency with the automatic generation system.

---

## Problem

### Incorrect (Manual Generation - Before Fix):

```
generated/content/social/
├── linkedin-article.json  ❌ (hyphen)
├── linkedin-post.json     ❌ (hyphen)
├── twitter.json           ✅
└── youtube.json           ✅
```

### Correct (Automatic Generation - Standard):

```
generated/content/social/
├── linkedin_article.json  ✅ (underscore)
├── linkedin_post.json     ✅ (underscore)
├── twitter.json           ✅
└── youtube.json           ✅
```

---

## Root Cause

The platform identifiers in the frontend use **hyphens** (`linkedin-article`, `linkedin-post`) for HTML compatibility, but the backend file names should use **underscores** (`linkedin_article.json`, `linkedin_post.json`) to match the existing automatic generation system.

---

## Solution

### Fixed Code in `app.py`:

**Before:**

```python
if 'linkedin-article' in platforms:
    # ...
    pipeline.append_json(os.path.join(output_dir, "linkedin-article.json"), data)
    #                                                      ^^^^^^^^^ WRONG
```

**After:**

```python
if 'linkedin-article' in platforms:
    # ...
    pipeline.append_json(os.path.join(output_dir, "linkedin_article.json"), data)
    #                                                      ^^^^^^^^^ CORRECT
```

---

## Changes Made

### 1. **app.py** - Fixed file names in `/generate_social_manual` endpoint:

- `linkedin-article.json` → `linkedin_article.json`
- `linkedin-post.json` → `linkedin_post.json`

### 2. **Deleted duplicate files:**

- Removed `generated/content/social/linkedin-article.json`
- Removed `generated/content/social/linkedin-post.json`

---

## Verification

### Correct File Structure Now:

```
generated/content/social/
├── linkedin_article.json  ✅ (Manual & Automatic both append here)
├── linkedin_post.json     ✅ (Manual & Automatic both append here)
├── linkedin.json          ✅ (Legacy - kept for backward compatibility)
├── twitter.json           ✅ (Manual & Automatic both append here)
└── youtube.json           ✅ (Manual & Automatic both append here)
```

---

## Key Takeaway

**Platform Identifiers vs File Names:**

- **Frontend/Platform IDs:** Use hyphens (`linkedin-article`, `linkedin-post`) for HTML/CSS compatibility
- **Backend/File Names:** Use underscores (`linkedin_article.json`, `linkedin_post.json`) for consistency with existing system

This ensures both manual and automatic generation append to the **same files**, maintaining data consistency.

---

## Testing Checklist

- [x] Manual generation creates correct file names (underscores)
- [x] Automatic generation still works (unchanged)
- [x] Both modes append to same files
- [x] No duplicate files created
- [x] Dashboard loads content correctly from underscore files
- [x] All platforms (LinkedIn Article, LinkedIn Post, Twitter, YouTube) work

---

**Fix Applied:** November 10, 2025  
**Status:** ✅ Resolved
