"""
Organize and rename Engine Rules PDFs for better clarity
"""
import os
import shutil

def organize_engine_rules():
    """Reorganize Engine Rules into a clear structure"""
    
    source_dir = "./Engine rules/Content Generation Rules"
    target_dir = "./Engine rules/01-Content-Generation"
    
    # Create target directories
    mimir_dir = os.path.join(target_dir, "MIMIR-System")
    parts_dir = os.path.join(target_dir, "Parts")
    
    os.makedirs(mimir_dir, exist_ok=True)
    os.makedirs(parts_dir, exist_ok=True)
    
    # File mapping: old name → new name
    file_mappings = {
        # MIMIR System files
        "defined.pdf": os.path.join(mimir_dir, "MIMIR-01-Defined.pdf"),
        "Control tower.pdf": os.path.join(mimir_dir, "MIMIR-02-Control-Tower.pdf"),
        "command set.pdf": os.path.join(mimir_dir, "MIMIR-03-Command-Set.pdf"),
        
        # Parts 1-16
        "1.pdf": os.path.join(parts_dir, "Part-01-Intent-Grounding.pdf"),
        "2.pdf": os.path.join(parts_dir, "Part-02-Phrasing-Foundations.pdf"),
        "3.pdf": os.path.join(parts_dir, "Part-03-Structure-Architecture.pdf"),
        "4.pdf": os.path.join(parts_dir, "Part-04-Tone-Decision-Engine.pdf"),
        "5.pdf": os.path.join(parts_dir, "Part-05-Anti-Patterns-Removal.pdf"),
        "6.pdf": os.path.join(parts_dir, "Part-06-Application-Logic.pdf"),
        "7.pdf": os.path.join(parts_dir, "Part-07-Tailoring-Principles.pdf"),
        "8.pdf": os.path.join(parts_dir, "Part-08-Structural-Tailoring.pdf"),
        "9.pdf": os.path.join(parts_dir, "Part-09-Cognitive-Bias-Counter.pdf"),
        "10.pdf": os.path.join(parts_dir, "Part-10-Evolution-Feedback.pdf"),
        "11.pdf": os.path.join(parts_dir, "Part-11-Semiotic-Safety.pdf"),
        "12.pdf": os.path.join(parts_dir, "Part-12-Narrative-Physics.pdf"),
        "13.pdf": os.path.join(parts_dir, "Part-13-Integrity-Grounding-Law.pdf"),
        "14.pdf": os.path.join(parts_dir, "Part-14-Visual-Orchestration.pdf"),
        "15.pdf": os.path.join(parts_dir, "Part-15-Logic-Emotion-Balance.pdf"),
        "16.pdf": os.path.join(parts_dir, "Part-16-Adaptive-Persona-Fusion.pdf"),
    }
    
    print("📁 Organizing Engine Rules...\n")
    
    # Copy and rename files
    copied_count = 0
    for old_name, new_path in file_mappings.items():
        old_path = os.path.join(source_dir, old_name)
        
        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            print(f"✅ {old_name} → {os.path.basename(new_path)}")
            copied_count += 1
        else:
            print(f"⚠️ Not found: {old_name}")
    
    print(f"\n📊 Summary:")
    print(f"   Files organized: {copied_count}/{len(file_mappings)}")
    print(f"   Target directory: {target_dir}")
    
    return target_dir

def create_index_file(target_dir):
    """Create an index file documenting all rules"""
    
    index_content = """# MIMIR Engine Rules Index

## Overview

This directory contains the complete MIMIR Messaging Humanoid Doctrine - a 16-part system for intelligent content generation with a Control Tower supervisory layer.

---

## Directory Structure

```
01-Content-Generation/
├── MIMIR-System/          # Core system definitions
│   ├── MIMIR-01-Defined.pdf
│   ├── MIMIR-02-Control-Tower.pdf
│   └── MIMIR-03-Command-Set.pdf
│
└── Parts/                 # 16-part content generation system
    ├── Part-01-Intent-Grounding.pdf
    ├── Part-02-Phrasing-Foundations.pdf
    └── ... (all 16 parts)
```

---

## MIMIR System Files

### MIMIR-01-Defined.pdf
**Purpose:** System overview and philosophy  
**Contains:**
- Mythic intelligence foundation
- Neural resonance principles
- Doctrinal fit and ecosystem integration
- Three-layer identity architecture

**Use When:** Understanding the overall MIMIR philosophy

---

### MIMIR-02-Control-Tower.pdf
**Purpose:** Master supervisory layer  
**Contains:**
- Coordination and enforcement rules
- 16-part execution sequence
- Contradiction resolution priorities
- Safety and validation protocols

**Use When:** Ensuring all parts work together correctly

---

### MIMIR-03-Command-Set.pdf
**Purpose:** Operational commands and directives  
**Contains:**
- System commands
- Execution instructions
- Mode selection guidelines

**Use When:** Implementing MIMIR in production

---

## The 16 Parts

### Part 01: Intent & Grounding
**File:** Part-01-Intent-Grounding.pdf  
**Purpose:** Clarify message intent, audience, risk level, emotional purpose  
**Use When:** Starting any content generation  
**Key Concepts:** Intent clarity, audience identification, risk assessment

---

### Part 02: Phrasing Foundations
**File:** Part-02-Phrasing-Foundations.pdf  
**Purpose:** Establish building blocks and phrasing architecture  
**Use When:** Structuring sentences and paragraphs  
**Key Concepts:** Sentence construction, word choice, clarity

---

### Part 03: Structure Architecture
**File:** Part-03-Structure-Architecture.pdf  
**Purpose:** Define content structure and flow  
**Use When:** Organizing content layout  
**Key Concepts:** Content hierarchy, flow, transitions

---

### Part 04: Tone Decision Engine
**File:** Part-04-Tone-Decision-Engine.pdf  
**Purpose:** Select correct tone mode, intensity, safety filters  
**Use When:** Determining voice and style  
**Key Concepts:** Professional, casual, bold tones; intensity levels; safety

---

### Part 05: Anti-Patterns Removal
**File:** Part-05-Anti-Patterns-Removal.pdf  
**Purpose:** Clean out weak phrasing, clichés, drift, mistakes  
**Use When:** Refining and polishing content  
**Key Concepts:** Cliché removal, clarity improvement, error correction

---

### Part 06: Application Logic
**File:** Part-06-Application-Logic.pdf  
**Purpose:** Check message fit, structure, technical correctness  
**Use When:** Validating content accuracy  
**Key Concepts:** Logical consistency, technical accuracy, fit

---

### Part 07: Tailoring Principles
**File:** Part-07-Tailoring-Principles.pdf  
**Purpose:** Apply persona, industry, role, region, language, seniority, culture  
**Use When:** Personalizing content for specific audiences  
**Key Concepts:** Persona matching, industry adaptation, cultural sensitivity

---

### Part 08: Structural Tailoring
**File:** Part-08-Structural-Tailoring.pdf  
**Purpose:** Shape message format to match intent and platform  
**Use When:** Adapting content for different platforms (blog, LinkedIn, Twitter)  
**Key Concepts:** Platform optimization, format adaptation

---

### Part 09: Cognitive Bias Counter-Logic
**File:** Part-09-Cognitive-Bias-Counter.pdf  
**Purpose:** Identify biases and correct cognitive errors  
**Use When:** Ensuring logical reasoning and fairness  
**Key Concepts:** Bias detection, logical fallacy correction

---

### Part 10: Evolution & Feedback Loops
**File:** Part-10-Evolution-Feedback.pdf  
**Purpose:** Self-check, correct, and refine  
**Use When:** Iterating and improving content  
**Key Concepts:** Self-correction, continuous improvement

---

### Part 11: Semiotic Safety
**File:** Part-11-Semiotic-Safety.pdf  
**Purpose:** Check meaning, symbolism, and subtext  
**Use When:** Ensuring cultural and symbolic appropriateness  
**Key Concepts:** Symbol safety, meaning accuracy, cultural respect

---

### Part 12: Narrative Physics
**File:** Part-12-Narrative-Physics.pdf  
**Purpose:** Check story logic, rhythm, pacing, and flow  
**Use When:** Creating narrative content  
**Key Concepts:** Story structure, pacing, narrative coherence

---

### Part 13: Integrity & Grounding Law
**File:** Part-13-Integrity-Grounding-Law.pdf  
**Purpose:** Eliminate drift, hallucination, unstated assumptions  
**Use When:** Ensuring factual accuracy and grounding  
**Key Concepts:** No hallucination, fact-checking, grounding

---

### Part 14: Visual Orchestration
**File:** Part-14-Visual-Orchestration.pdf  
**Purpose:** Convert text to accurate/cinematic visuals  
**Use When:** Generating image or video prompts  
**Key Concepts:** Faithful vs cinematic modes, visual continuity, safety

---

### Part 15: Logic-Emotion Balance
**File:** Part-15-Logic-Emotion-Balance.pdf  
**Purpose:** Balance logic with emotion, embed consequence stacking  
**Use When:** Creating persuasive content  
**Key Concepts:** Emotional voltage, logic-emotion ratio, consequence framing

---

### Part 16: Adaptive Persona Fusion
**File:** Part-16-Adaptive-Persona-Fusion.pdf  
**Purpose:** Combine all persona signals into coherent model  
**Use When:** Final persona alignment  
**Key Concepts:** Persona synthesis, signal integration

---

## Usage Guidelines

### For Blog Generation
**Required Parts:** 1, 2, 3, 4, 7, 12, 13, 15  
**Optional Parts:** 5, 6, 10, 11

### For Social Media (LinkedIn/Twitter)
**Required Parts:** 1, 4, 7, 8, 15  
**Optional Parts:** 5, 11, 12

### For Visual Prompts
**Required Parts:** 14  
**Supporting Parts:** 1, 11, 13

### For ICP Extraction
**Required Parts:** 1, 6, 7, 13  
**Supporting Parts:** 9, 10

### For Quality Scoring
**Required Parts:** 5, 6, 10, 13  
**Supporting Parts:** 4, 11, 12, 15

---

## Control Tower Execution Sequence

The Control Tower enforces this rigid execution order:

1. **Grounding & Intent** (Part 1)
2. **Phrasing & Structure** (Parts 2-3)
3. **Tone Decision** (Part 4)
4. **Anti-Patterns Removal** (Part 5)
5. **Application Logic** (Part 6)
6. **Tailoring Principles** (Part 7)
7. **Structural Tailoring** (Part 8)
8. **Cognitive Bias Counter** (Part 9)
9. **Evolution & Feedback** (Part 10)
10. **Semiotic Safety** (Part 11)
11. **Narrative Physics** (Part 12)
12. **Integrity & Grounding** (Part 13)
13. **Visual Orchestration** (Part 14) - if needed
14. **Logic-Emotion Balance** (Part 15)
15. **Adaptive Persona Fusion** (Part 16)
16. **Final Validation** (Control Tower)

---

## Priority Hierarchy (for Contradictions)

When rules conflict, the Control Tower resolves using this priority order:

1. **Safety** (identity, cultural, emotional, context, symbolic, narrative)
2. **User Intent** (preserve intended outcome)
3. **Persona Protection** (dignity, no mismatch, no overwhelm)
4. **Platform Compatibility** (natural rhythm fit)
5. **Narrative Fidelity** (story coherence)
6. **Structural Logic** (persona shape alignment)
7. **Emotional Voltage Fit** (safety and readiness match)
8. **CTA Accuracy** (authority and decision power match)

---

## Integration with Marketing Engine

### Current Integration Points:
- **Blog Generator** (`blog_generator.py`)
- **Social Media Generator** (`post_generator.py`)
- **Image Prompt Builder** (`image_prompt_builder.py`)

### Future Integration Points:
- **ICP Extractor** (`extractor.py`)
- **Quality Scorer** (quality scoring system)
- **Trend Analyzer** (`trend_fetcher.py`)

---

## Version History

- **v1.0** - Initial organization (November 2025)
- Complete 16-part system with Control Tower
- 19 PDF files organized into logical structure

---

## Maintenance Notes

- Keep PDFs separate for granular vector search
- Update individual parts as needed
- Maintain this index when adding new rules
- Test changes with sample content generation

---

**Last Updated:** November 14, 2025  
**Total Files:** 19 PDFs (3 MIMIR system + 16 parts)  
**Total Size:** ~550KB
"""
    
    index_path = os.path.join(target_dir, "INDEX.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"\n📄 Created index file: {index_path}")
    return index_path

if __name__ == "__main__":
    print("🚀 Starting Engine Rules Organization...\n")
    
    # Organize files
    target_dir = organize_engine_rules()
    
    # Create index
    index_path = create_index_file(target_dir)
    
    print("\n✅ Organization complete!")
    print(f"\n📂 New structure:")
    print(f"   {target_dir}/")
    print(f"   ├── MIMIR-System/ (3 files)")
    print(f"   ├── Parts/ (16 files)")
    print(f"   └── INDEX.md")
    
    print(f"\n📖 Next steps:")
    print(f"   1. Review the organized files in: {target_dir}")
    print(f"   2. Read INDEX.md for complete documentation")
    print(f"   3. Run engine_kb_builder.py to create vector database")
