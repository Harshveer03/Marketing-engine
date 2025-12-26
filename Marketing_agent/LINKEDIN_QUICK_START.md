# LinkedIn Content Guide - Quick Start

## 🚀 3-Step Setup

### **Step 1: Build Engine KB**
```bash
python engine_kb_builder.py
```

### **Step 2: Test Integration**
```bash
python test_mimir_integration.py
```

### **Step 3: Generate Content**
```bash
python app.py
```

---

## 📝 What You Get

### **LinkedIn Posts (200-300 words)**

**Structure:**
```
Hook (1 punchy line)
    ↓
Context (2-4 lines setting the scene)
    ↓
Insight (core truth)
    ↓
Story (specific example)
    ↓
Consequence (what happens if ignored)
    ↓
Shift (one actionable change)
    ↓
Close (reflective question)
    ↓
Hashtags (5-7)
```

**Example Output:**
```
Founder-led sales breaks at scale.

Not because founders aren't good at sales.
Because they can't clone themselves.

The real problem? You're treating it like a process issue.
It's a behavior rewiring challenge.

I've seen teams hire 5 AEs, give them the "founder playbook," 
and watch conversion drop 40%.

The cost isn't just lost deals.
It's buyer confusion. Rep anxiety. Pipeline unpredictability.

The fix isn't more coaching.
It's systematic execution rewiring at the moment level.

Are you scaling a process or rewiring behavior?

#B2BSaaS #GTM #SalesExecution
```

### **LinkedIn Articles (500-600 words)**

Same 7-section structure but expanded with:
- Detailed paragraphs
- Multiple examples
- Comprehensive analysis
- Actionable frameworks

---

## 🎨 Post Types (Auto-Selected)

| Tone | Post Type | Use For |
|------|-----------|---------|
| Sharp | Jolt | Wake-up calls, harsh truths |
| Reflective | Narrative | Personal stories, lessons learned |
| Teaching | Teaching | How-to, skill improvement |
| Contrarian | Contrarian | Challenging beliefs |
| Analytical | Insight | Pattern explanation |

---

## 🎯 50 Skeletons (Auto-Selected)

System picks from 50 content patterns:

- **Hidden Cost** - Unseen losses
- **Everyone Gets This Wrong** - Common mistakes
- **Bought Lesson** - Expensive mistakes
- **Daily Blind Spot** - Overlooked issues
- **One Moment That Changes Everything** - Turning points
- **Why You Feel Stuck** - Blockages
- **What You Think vs What Actually Happens** - Reality gaps
- **Quiet Skill That Changes Outcomes** - Underrated skills
- **Nobody Tells You This Early** - Late learnings
- **Stop Doing X If You Want Y** - Tradeoffs
- ... (40 more)

---

## 🔍 How It Works

```
Your Input:
- Topic: "AI in sales execution"
- Tone: "sharp"
- Persona: "CXO"
- Industry: "B2B SaaS"

    ↓

System Automatically:
1. Selects Post Type: "Jolt" (sharp tone)
2. Selects Skeleton: "Everyone Gets This Wrong"
3. Applies Template: T_JOLT
4. Follows Section Prompts
5. Adapts for CXO + B2B SaaS
6. Generates Structured Content

    ↓

Output:
Structured LinkedIn post following proven framework
```

---

## ✅ Success Indicators

Look for these in console:

```
============================================================
🎯 DEEP LINKEDIN INTEGRATION
============================================================

🔍 LinkedIn Content Guide Query:
   Content Type: post
   Tone: sharp
   Persona: CXOs
   Industry: B2B SaaS
   Topic: AI in sales execution...

📚 Retrieved 8 LinkedIn Guide chunks
✅ LinkedIn Content Guide structure: 4521 characters
✅ MIMIR rules loaded: 2847 chars
============================================================
```

---

## 🎓 Tone Guide

### **Sharp**
- Direct, punchy
- Consequence-focused
- Wake-up call style
- Example: "Founder-led sales breaks at scale."

### **Reflective**
- Thoughtful, introspective
- Story-driven
- Lesson-focused
- Example: "The moment I realized coaching wasn't enough..."

### **Teaching**
- Clear, practical
- Step-by-step
- Skill-focused
- Example: "Here's how to fix shallow discovery calls..."

### **Contrarian**
- Bold, challenging
- Belief-flipping
- Provocative
- Example: "Everyone says hire more reps. They're wrong."

### **Neutral/Professional**
- Balanced, accessible
- Data-driven
- Analytical
- Example: "Three patterns in successful GTM teams..."

---

## 📊 Quality Checklist

Before publishing, verify:

- [ ] Hook grabs attention (first line)
- [ ] Context sets the scene (2-4 lines)
- [ ] Insight is clear and valuable
- [ ] Story is specific and relatable
- [ ] Consequence shows real impact
- [ ] Shift offers actionable change
- [ ] Close invites reflection
- [ ] Tone is consistent throughout
- [ ] Language matches persona
- [ ] Examples fit industry
- [ ] 5-7 relevant hashtags

---

## 🛠️ Troubleshooting

### **Generic posts still generated?**
```bash
# Rebuild Engine KB
python engine_kb_builder.py

# Verify LinkedIn Guide is included
ls "Engine rules/01-Content-Generation/Linkedin Content Guide.pdf"
```

### **Wrong post type selected?**
Be more explicit with tone:
- "sharp" → Jolt
- "reflective" → Narrative
- "teaching" → Teaching
- "contrarian" → Contrarian

### **Structure not followed?**
Check console for LinkedIn Guide query messages.
If missing, Engine KB needs rebuilding.

---

## 📚 Full Documentation

- **Comprehensive Guide:** `LINKEDIN_GUIDE_INTEGRATION.md`
- **Integration Summary:** `LINKEDIN_DEEP_INTEGRATION_SUMMARY.md`
- **MIMIR System:** `MIMIR_INTEGRATION_SUMMARY.md`
- **Quick Reference:** `MIMIR_QUICK_REFERENCE.md`

---

## 🎉 You're Ready!

The system now generates structured, high-quality LinkedIn content following a proven framework with:

✅ 5 post types  
✅ 50 content skeletons  
✅ 7-section structure  
✅ Tone-adapted language  
✅ Persona-specific examples  
✅ Industry-relevant context  

**Start generating amazing LinkedIn content!** 🚀

---

**Version:** 1.0  
**Last Updated:** November 14, 2025  
**Status:** Production Ready
