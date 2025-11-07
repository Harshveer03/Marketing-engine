import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class ImagePromptBuilder:
    def __init__(self, model="models/gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
    
    def blog_image_prompt(self, blog_data):
        """
        Generate an optimized image prompt for a blog post.
        
        Args:
            blog_data (dict): Blog information containing:
                - title (str): Blog title
                - blog (str): Full blog content
                - outline (list): Blog outline/sections
                - industry (str): Target industry
                - tone (str): Content tone
                - audience (str): Target audience
        
        Returns:
            str: Optimized image generation prompt
        """
        # Extract key information
        title = blog_data.get('title', '')
        content = blog_data.get('blog', '')
        outline = blog_data.get('outline', [])
        industry = blog_data.get('industry', 'B2B SaaS')
        tone = blog_data.get('tone', 'professional')
        audience = blog_data.get('audience', 'CXOs')
        
        # Extract key themes from content (first 1000 characters for context)
        content_preview = content[:1000] if content else ''
        
        # Build the prompt for AI to generate image prompt
        prompt = f"""
        You are an expert visual content strategist specializing in creating optimized image generation prompts for AI image generators like DALL-E, Midjourney, and Stable Diffusion.

        Create a detailed, optimized image generation prompt for a blog post with the following details:

        Blog Title: "{title}"
        Industry: {industry}
        Target Audience: {audience}
        Tone: {tone}
        Key Sections: {', '.join(outline) if outline else 'N/A'}
        Content Preview: {content_preview}

        Requirements for the image prompt:
        1. Professional, editorial-style imagery suitable for a blog header/featured image
        2. Reflects the blog's core theme and {industry} industry context
        3. Appropriate for {audience} audience - sophisticated and authoritative
        4. Tone should be {tone}
        5. High-quality, modern aesthetic
        6. Suitable for 1200x630px blog header format
        7. Avoid text in the image
        8. Focus on visual metaphors and concepts, not literal representations
        9. Use corporate/professional color schemes appropriate for {industry}
        10. Should convey thought leadership and expertise

        Image Style Guidelines:
        - For "professional" tone: Clean, minimalist, corporate aesthetic
        - For "bold" tone: Dynamic, vibrant, attention-grabbing
        - For "casual" tone: Approachable, friendly, modern
        - For IT & Dev: Tech-forward, digital, data-driven visuals
        - For Sales/Marketing: Growth-focused, strategic, people-centric
        - For Real Estate: Property-focused, architectural, professional

        Generate a single, comprehensive image prompt (150-200 words) that can be directly used in an AI image generator.
        The prompt should be detailed, specific, and optimized for high-quality results.

        Return ONLY the image prompt text, no explanations or additional commentary.
        """
        
        try:
            # Generate the image prompt using AI
            response = self.llm.invoke(prompt).content
            
            # Clean up the response
            image_prompt = response.strip()
            
            # Remove any markdown formatting or quotes
            image_prompt = re.sub(r'^["\'`]+|["\'`]+$', '', image_prompt)
            image_prompt = re.sub(r'^\*\*|\*\*$', '', image_prompt)
            
            # Add quality enhancers if not present
            quality_keywords = ['high quality', 'professional', '4k', '8k', 'detailed']
            if not any(keyword in image_prompt.lower() for keyword in quality_keywords):
                image_prompt += ", high quality, professional photography, 4k resolution"
            
            print(f"✅ Generated image prompt for blog: '{title[:50]}...'")
            return image_prompt
            
        except Exception as e:
            print(f"❌ Error generating image prompt: {e}")
            # Return a fallback prompt
            return self._generate_fallback_blog_prompt(title, industry, tone, audience)
    
    def _generate_fallback_blog_prompt(self, title, industry, tone, audience):
        """Generate a fallback image prompt if AI generation fails"""
        
        # Industry-specific visual elements
        industry_visuals = {
            'IT & Dev': 'modern technology dashboard, code visualization, digital infrastructure',
            'Sales/Marketing': 'business growth charts, team collaboration, strategic planning',
            'Real Estate': 'modern architecture, property development, urban landscape',
            'Finance': 'financial data visualization, market analysis, investment strategy',
            'Healthcare': 'medical technology, healthcare innovation, patient care',
            'B2B SaaS': 'software interface, cloud technology, digital transformation'
        }
        
        # Tone-specific aesthetics
        tone_aesthetics = {
            'professional': 'clean minimalist design, corporate color palette, sophisticated composition',
            'bold': 'vibrant colors, dynamic composition, eye-catching design',
            'casual': 'friendly approachable style, warm colors, modern aesthetic',
            'technical': 'precise detailed illustration, technical diagrams, analytical style'
        }
        
        # Audience-specific elements
        audience_elements = {
            'CXOs': 'executive boardroom aesthetic, strategic vision, leadership perspective',
            'Founders': 'entrepreneurial energy, innovation focus, startup culture',
            'Marketers': 'creative campaign visuals, brand storytelling, audience engagement',
            'Developers': 'code-centric design, technical precision, developer tools'
        }
        
        visual_element = industry_visuals.get(industry, industry_visuals['B2B SaaS'])
        aesthetic = tone_aesthetics.get(tone, tone_aesthetics['professional'])
        audience_element = audience_elements.get(audience, audience_elements['CXOs'])
        
        fallback_prompt = f"""Professional editorial illustration for blog titled '{title}', 
        featuring {visual_element}, {aesthetic}, {audience_element}, 
        modern {industry} industry aesthetic, suitable for thought leadership content, 
        no text overlay, corporate professional style, high quality 4k resolution, 
        clean composition, suitable for blog header image 1200x630px format"""
        
        # Clean up extra whitespace
        fallback_prompt = ' '.join(fallback_prompt.split())
        
        return fallback_prompt


# Standalone function for easy import
def blog_image_prompt(blog_data):
    """
    Convenience function to generate blog image prompt.
    
    Args:
        blog_data (dict): Blog information
    
    Returns:
        str: Optimized image generation prompt
    """
    builder = ImagePromptBuilder()
    return builder.blog_image_prompt(blog_data)


# Test function
if __name__ == "__main__":
    # Test with sample blog data
    sample_blog = {
        "title": "AI rewires sales execution: Drive immediate revenue impact",
        "blog": "Introduction: The Imperative of Rewiring Sales for the AI Era. The B2B SaaS landscape is characterized by relentless innovation...",
        "outline": [
            "Introduction: The Imperative of Rewiring Sales for the AI Era",
            "The Cracks in Traditional GTM for IT & Dev SaaS",
            "AI as the GTM Behavior Rewiring System",
            "Conclusion: Seize the AI-Driven Revenue Advantage"
        ],
        "industry": "IT & Dev",
        "tone": "professional",
        "audience": "CXOs"
    }
    
    print("\n--- Testing Blog Image Prompt Generator ---\n")
    prompt = blog_image_prompt(sample_blog)
    print(f"\nGenerated Prompt:\n{prompt}\n")
