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


    def linkedin_image_prompt(self, linkedin_data):
        """
        Generate an optimized image prompt for a LinkedIn post.
        
        Args:
            linkedin_data (dict): LinkedIn post information containing:
                - title (str): Post title
                - caption (str): Post caption/content
                - hashtags (list): Post hashtags
                - industry (str): Target industry
                - tone (str): Content tone
                - audience (str): Target audience
        
        Returns:
            str: Optimized image generation prompt
        """
        # Extract key information
        title = linkedin_data.get('title', '')
        caption = linkedin_data.get('caption', '')
        hashtags = linkedin_data.get('hashtags', [])
        industry = linkedin_data.get('industry', 'B2B SaaS')
        tone = linkedin_data.get('tone', 'professional')
        audience = linkedin_data.get('audience', 'CXOs')
        
        # Extract key themes from caption (first 500 characters for context)
        caption_preview = caption[:500] if caption else ''
        
        # Build the prompt for AI to generate image prompt
        prompt = f"""
        You are an expert visual content strategist specializing in creating optimized image generation prompts for AI image generators like DALL-E, Midjourney, and Stable Diffusion.

        Create a detailed, optimized image generation prompt for a LinkedIn post with the following details:

        Post Title: "{title}"
        Industry: {industry}
        Target Audience: {audience}
        Tone: {tone}
        Hashtags: {', '.join(hashtags) if hashtags else 'N/A'}
        Caption Preview: {caption_preview}

        Requirements for the image prompt:
        1. Professional, corporate-style imagery suitable for LinkedIn feed (1200x627px format)
        2. Reflects the post's core message and {industry} industry context
        3. Appropriate for {audience} audience - sophisticated and business-focused
        4. Tone should be {tone}
        5. High-quality, modern aesthetic optimized for social media engagement
        6. Suitable for LinkedIn's professional environment
        7. Avoid text in the image (LinkedIn will overlay the caption)
        8. Focus on visual metaphors that convey thought leadership
        9. Use corporate/professional color schemes appropriate for {industry}
        10. Should drive engagement and shares on LinkedIn

        Image Style Guidelines:
        - For "professional" tone: Clean, corporate aesthetic, business-focused
        - For "bold" tone: Dynamic, attention-grabbing, confident
        - For "casual" tone: Approachable yet professional, modern
        - For IT & Dev: Tech-forward, digital innovation, data visualization
        - For Sales/Marketing: Growth charts, team collaboration, strategic vision
        - For LinkedIn: Professional networking aesthetic, thought leadership visual

        Generate a single, comprehensive image prompt (150-200 words) that can be directly used in an AI image generator.
        The prompt should be detailed, specific, and optimized for LinkedIn engagement.

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
            
            print(f"✅ Generated LinkedIn image prompt for post: '{title[:50]}...'")
            return image_prompt
            
        except Exception as e:
            print(f"❌ Error generating LinkedIn image prompt: {e}")
            # Return a fallback prompt
            return self._generate_fallback_linkedin_prompt(title, industry, tone, audience)
    
    def _generate_fallback_linkedin_prompt(self, title, industry, tone, audience):
        """Generate a fallback LinkedIn image prompt if AI generation fails"""
        
        # Industry-specific visual elements for LinkedIn
        industry_visuals = {
            'IT & Dev': 'modern technology workspace, digital innovation, tech professionals collaborating',
            'Sales/Marketing': 'business growth visualization, strategic planning session, professional team meeting',
            'Real Estate': 'modern office building, professional property showcase, business district',
            'Finance': 'financial analytics dashboard, professional investment meeting, market data visualization',
            'Healthcare': 'healthcare innovation, medical professionals, modern healthcare facility',
            'B2B SaaS': 'software collaboration, cloud technology, professional digital workspace'
        }
        
        # Tone-specific aesthetics for LinkedIn
        tone_aesthetics = {
            'professional': 'clean corporate design, business professional aesthetic, sophisticated composition',
            'bold': 'dynamic business visuals, confident professional style, eye-catching corporate design',
            'casual': 'approachable professional style, modern business aesthetic, friendly corporate environment',
            'technical': 'precise business illustration, professional technical diagrams, analytical corporate style'
        }
        
        # Audience-specific elements for LinkedIn
        audience_elements = {
            'CXOs': 'executive leadership aesthetic, C-suite perspective, strategic business vision',
            'Founders': 'entrepreneurial innovation, startup leadership, business growth focus',
            'Marketers': 'marketing strategy visuals, brand leadership, professional campaign aesthetic',
            'Developers': 'technical innovation, professional development environment, tech leadership'
        }
        
        visual_element = industry_visuals.get(industry, industry_visuals['B2B SaaS'])
        aesthetic = tone_aesthetics.get(tone, tone_aesthetics['professional'])
        audience_element = audience_elements.get(audience, audience_elements['CXOs'])
        
        fallback_prompt = f"""Professional LinkedIn post image for '{title}', 
        featuring {visual_element}, {aesthetic}, {audience_element}, 
        modern {industry} industry aesthetic, suitable for thought leadership content on LinkedIn, 
        no text overlay, corporate professional style optimized for social media engagement, 
        high quality 4k resolution, clean composition, suitable for LinkedIn feed image 1200x627px format"""
        
        # Clean up extra whitespace
        fallback_prompt = ' '.join(fallback_prompt.split())
        
        return fallback_prompt
    
    def twitter_image_prompt(self, twitter_data):
        """
        Generate an optimized image prompt for a Twitter/X post.
        
        Args:
            twitter_data (dict): Twitter post information containing:
                - title (str): Post title
                - caption (str): Tweet text
                - hashtags (list): Post hashtags
                - industry (str): Target industry
                - tone (str): Content tone
                - audience (str): Target audience
        
        Returns:
            str: Optimized image generation prompt
        """
        # Extract key information
        title = twitter_data.get('title', '')
        caption = twitter_data.get('caption', '')
        hashtags = twitter_data.get('hashtags', [])
        industry = twitter_data.get('industry', 'B2B SaaS')
        tone = twitter_data.get('tone', 'professional')
        audience = twitter_data.get('audience', 'CXOs')
        
        # Twitter content is already concise
        tweet_text = caption[:280] if caption else ''
        
        # Build the prompt for AI to generate image prompt
        prompt = f"""
        You are an expert visual content strategist specializing in creating optimized image generation prompts for AI image generators like DALL-E, Midjourney, and Stable Diffusion.

        Create a detailed, optimized image generation prompt for a Twitter/X post with the following details:

        Post Title: "{title}"
        Industry: {industry}
        Target Audience: {audience}
        Tone: {tone}
        Hashtags: {', '.join(hashtags) if hashtags else 'N/A'}
        Tweet Text: {tweet_text}

        Requirements for the image prompt:
        1. Eye-catching, scroll-stopping imagery suitable for Twitter feed (1200x675px format)
        2. Reflects the tweet's core message and {industry} industry context
        3. Appropriate for {audience} audience - professional yet engaging
        4. Tone should be {tone}
        5. High-quality, modern aesthetic optimized for Twitter engagement and retweets
        6. Suitable for Twitter's fast-paced, visual-first environment
        7. Avoid text in the image (Twitter will show the tweet text)
        8. Focus on bold, clear visual concepts that grab attention quickly
        9. Use vibrant, engaging colors appropriate for {industry} while maintaining professionalism
        10. Should drive engagement, retweets, and replies

        Image Style Guidelines:
        - For "professional" tone: Clean, modern, business-focused but engaging
        - For "bold" tone: Dynamic, vibrant, attention-grabbing, confident
        - For "casual" tone: Approachable, friendly, conversational yet professional
        - For IT & Dev: Tech-forward, digital, innovative, cutting-edge
        - For Sales/Marketing: Growth-focused, strategic, results-driven
        - For Twitter: Fast-paced, engaging, shareable, conversation-starting visual

        Generate a single, comprehensive image prompt (150-200 words) that can be directly used in an AI image generator.
        The prompt should be detailed, specific, and optimized for Twitter engagement.

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
            
            print(f"✅ Generated Twitter image prompt for post: '{title[:50]}...'")
            return image_prompt
            
        except Exception as e:
            print(f"❌ Error generating Twitter image prompt: {e}")
            # Return a fallback prompt
            return self._generate_fallback_twitter_prompt(title, industry, tone, audience)
    
    def _generate_fallback_twitter_prompt(self, title, industry, tone, audience):
        """Generate a fallback Twitter image prompt if AI generation fails"""
        
        # Industry-specific visual elements for Twitter
        industry_visuals = {
            'IT & Dev': 'modern tech innovation, digital transformation, cutting-edge technology',
            'Sales/Marketing': 'business growth metrics, marketing success, strategic wins',
            'Real Estate': 'modern architecture, property innovation, urban development',
            'Finance': 'financial success, market trends, investment insights',
            'Healthcare': 'healthcare innovation, medical breakthroughs, patient care excellence',
            'B2B SaaS': 'software innovation, digital solutions, tech transformation'
        }
        
        # Tone-specific aesthetics for Twitter
        tone_aesthetics = {
            'professional': 'clean modern design, professional aesthetic, engaging composition',
            'bold': 'vibrant dynamic visuals, bold confident style, attention-grabbing design',
            'casual': 'friendly approachable style, conversational aesthetic, modern engaging design',
            'technical': 'precise technical illustration, innovative diagrams, analytical style'
        }
        
        # Audience-specific elements for Twitter
        audience_elements = {
            'CXOs': 'executive insights, leadership perspective, strategic vision',
            'Founders': 'entrepreneurial energy, startup innovation, growth mindset',
            'Marketers': 'marketing insights, creative strategy, campaign excellence',
            'Developers': 'technical innovation, developer insights, coding excellence'
        }
        
        visual_element = industry_visuals.get(industry, industry_visuals['B2B SaaS'])
        aesthetic = tone_aesthetics.get(tone, tone_aesthetics['professional'])
        audience_element = audience_elements.get(audience, audience_elements['CXOs'])
        
        fallback_prompt = f"""Engaging Twitter post image for '{title}', 
        featuring {visual_element}, {aesthetic}, {audience_element}, 
        modern {industry} industry aesthetic, suitable for thought leadership content on Twitter, 
        no text overlay, professional yet engaging style optimized for social media virality, 
        high quality 4k resolution, eye-catching composition, suitable for Twitter feed image 1200x675px format"""
        
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


def linkedin_image_prompt(linkedin_data):
    """
    Convenience function to generate LinkedIn image prompt.
    
    Args:
        linkedin_data (dict): LinkedIn post information
    
    Returns:
        str: Optimized image generation prompt
    """
    builder = ImagePromptBuilder()
    return builder.linkedin_image_prompt(linkedin_data)


def twitter_image_prompt(twitter_data):
    """
    Convenience function to generate Twitter image prompt.
    
    Args:
        twitter_data (dict): Twitter post information
    
    Returns:
        str: Optimized image generation prompt
    """
    builder = ImagePromptBuilder()
    return builder.twitter_image_prompt(twitter_data)


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
