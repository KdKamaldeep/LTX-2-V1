#!/usr/bin/env python3
"""
Script to generate LTX-2 video generation prompts as a short film script using OpenAI API.
Generates cohesive scene prompts that form a complete narrative, following LTX-2 best practices.

The script:
1. Asks for a story title
2. Asks for a story description
3. Generates a story outline
4. Creates scene-by-scene prompts that flow together as a short film script
5. Outputs prompts in JSON format

Requirements:
    pip install openai

Usage:
    python generate_ltx2_prompts.py
"""

import json
import argparse
from openai import OpenAI
from typing import List, Dict

# OpenAI API Key
OPENAI_API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def create_prompt_generation_instruction() -> str:
    """Creates the instruction prompt for OpenAI to generate LTX-2 prompts."""
    return """You are an expert video prompt writer and screenwriter specializing in LTX-2 video generation. 
    You create cohesive short film scripts where each prompt represents a scene that flows naturally into the next.
    
    PROMPT FORMAT AND STYLE:
    - You may use screenplay format (INT./EXT. LOCATION – TIME) when appropriate, especially for establishing shots
    - Include detailed dialogue with character attribution: "Character (tone/emotion): 'dialogue text'"
    - When a specific style is provided (e.g., "pixar style", "sci-fi style", "film noir"), explicitly mention it in the prompt
    - Use present tense throughout
    - Write in a flowing narrative style that reads like a film script
    
    KEY ASPECTS TO INCLUDE IN EACH SCENE PROMPT:
    1. Establish the shot - Use cinematography terms matching film genres. Include scale and category characteristics.
       Examples: "Cinematic action packed shot", "Animated cinematic shot", "A warm, intimate cinematic performance"
    2. Set the scene - Describe lighting conditions, color palette, surface textures, and atmosphere in detail.
       Examples: "Warm sunny backyard", "cozy, wood-paneled bar, lit with soft amber practical lights", "dark lit room"
    3. Describe the action - Write core action as a natural sequence, flowing from beginning to end with specific timing and beats.
       Include pauses, beats, and dramatic timing: "A beat.", "Beat.", "Quick zoom back", "Cut to side view"
    4. Define character(s) - Include age, ethnicity (when relevant), hairstyle, clothing, and distinguishing details. 
       Express emotions through physical cues and body language.
       Examples: "young african american woman wearing a futuristic transparent visor", "woman and a man in their 30s", 
       "senior frog instructor", "young female singer in her 20s with short brown hair and bangs"
    5. Identify camera movement(s) - Specify exact camera movements with precise terminology.
       Examples: "handheld tracking", "crane up", "dolly back", "zoom in", "pan right", "slow dolly in", 
       "camera arcs left", "camera zooms in on", "camera pans to reveal", "hand held feel to the camera"
    6. Describe the audio - Use clear descriptions for ambient sounds, music, and speech. 
       For dialogue, use quotation marks and include character attribution and tone.
       Examples: "says softly", "whispering dramatically", "says with an angry african american accent", 
       "says with a low robotic voice", "mouth full", "off-screen, shouting over the noise"
    7. Include style specification - When a style is provided, explicitly mention it in the prompt.
       Examples: "pixar style acting and timing", "sci-fi style cinematic scene", "Cinematic action packed shot"

    FOR BEST RESULTS:
    - Use specific camera language: "handheld tracking", "crane up", "dolly back", "zoom in/out", "pan left/right", "arc around"
    - Include detailed cinematography: depth of field, bokeh, lighting quality, color temperature
    - Describe motion blur, dust, steam, and atmospheric effects when relevant
    - Include dramatic timing: "A beat.", "Beat.", pauses, "then", "suddenly"
    - Match detail to shot scale (closeups need more detail than wide shots)
    - Write 4 to 12 descriptive sentences covering all key aspects
    - Each scene should advance the story naturally
    - Maintain consistency in characters, setting, and tone throughout
    - When style is specified, ensure it's mentioned explicitly in the prompt

    STYLE EXAMPLES TO FOLLOW:
    - "pixar style acting and timing" - Include exaggerated expressions, comedic timing, character animation style
    - "sci-fi style cinematic scene" - Futuristic elements, advanced technology, space-age aesthetics
    - "film noir" - High contrast lighting, shadows, dramatic angles, period-appropriate styling
    - "animated cinematic shot" - Animation-specific camera movements and character design
    - "Cinematic action packed shot" - Dynamic camera work, motion blur, intense energy

    CATEGORIES TO USE:
    Animation: stop-motion, 2D/3D animation, claymation, hand-drawn, Studio Ghibli style, Pixar style
    Stylized: comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated, Studio Ghibli aesthetic
    Cinematic: period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary, sci-fi

    WHAT WORKS WELL:
    - Cinematic compositions with thoughtful lighting (golden hour, warm amber, harsh shadows, soft practical lights)
    - Emotive human moments and facial nuance
    - Weather effects (fog, mist, golden hour, rain, steam)
    - Clear camera language with specific movements ("slow dolly in", "handheld tracking", "crane up", "zoom in on")
    - Stylized aesthetics (painterly, noir, analog film look, Studio Ghibli style, Pixar style)
    - Detailed dialogue with character attribution and tone
    - Screenplay-style scene headers when appropriate (INT./EXT. LOCATION – TIME)
    - Motion effects (motion blur, dust, steam, reflections)
    - Depth of field and bokeh descriptions
    - Dramatic timing and beats

    WHAT TO AVOID:
    - Emotional labels without visual cues (use posture/gesture instead)
    - Text, logos, or brand names
    - Complex physics or chaotic motion
    - Too many characters or excessive objects
    - Inconsistent lighting logic
    - Overly complicated prompts
    - Forgetting to mention the specified style when one is provided

    Create scene prompts that form a cohesive narrative following ALL these guidelines, matching the detailed, cinematic style of professional LTX-2 prompts."""


def generate_story_outline(story_title: str, story_description: str, num_scenes: int, style: str = "", model: str = "gpt-4o-mini") -> str:
    """
    Generate a story outline for the short film.
    
    Args:
        story_title: Title of the short film
        story_description: Description of the story to provide context
        num_scenes: Number of scenes in the script
        style: Visual/cinematic style of the film
        model: OpenAI model to use
    
    Returns:
        Story outline as a string
    """
    system_prompt = """You are an expert screenwriter. Create a brief story outline for a youtube short film.
    The outline should describe the narrative flow and what happens in each scene, ensuring they connect logically."""
    
    style_context = f" The film should be in {style} style." if style else ""
    
    description_context = f"\n\nStory Description:\n{story_description}" if story_description else ""
    
    user_message = f"""Create a brief story outline for a short film titled "{story_title}".
    The film should have {num_scenes} scenes that form a complete narrative arc.{style_context}{description_context}
    Provide a concise outline describing what happens in each scene and how they connect.
    Keep it to 2-3 sentences per scene."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠ Warning: Could not generate story outline: {str(e)}")
        return ""


def generate_ltx2_prompts(story_title: str, story_description: str, num_scenes: int, story_outline: str, style: str = "", model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    """
    Generate LTX-2 video prompts as scenes for a short film script in a single API call.
    
    Args:
        story_title: Title of the short film
        story_description: Description of the story to provide context
        num_scenes: Number of scene prompts to generate
        story_outline: The story outline to maintain narrative consistency
        style: Visual/cinematic style of the film
        model: OpenAI model to use (default: gpt-4o-mini, supports JSON mode)
    
    Returns:
        List of dictionaries with 'title' (scene title) and 'prompt' keys
    """
    system_prompt = create_prompt_generation_instruction()
    
    style_context = f"\n\nIMPORTANT - Visual Style: {style}\nYou MUST explicitly mention this style in each scene prompt (e.g., '{style} style', '{style} cinematic scene', '{style} acting and timing'). Apply this style consistently throughout all scenes, incorporating appropriate cinematography, lighting, color palette, and aesthetic elements that match this style. The style should be mentioned naturally within the prompt text, not just as a tag." if style else ""
    
    description_context = f"\n\nStory Description:\n{story_description}" if story_description else ""
    
    user_message = f"""You are writing a complete short youtube film script with {num_scenes} scenes for a film titled "{story_title}".

Story Outline:
{story_outline}{description_context}{style_context}

Generate ALL {num_scenes} scenes that:
- Follow the story outline and advance the narrative from beginning to end
- Each scene builds naturally from the previous one, creating a cohesive narrative flow
- Maintain consistency with characters, setting, and tone throughout
- Each scene is complete and cinematic, following LTX-2 prompt guidelines
- Scene 1 establishes the opening, middle scenes develop the story, final scene provides resolution
- Include detailed camera movements, dialogue with character attribution, and specific cinematography descriptions
- When style is specified, explicitly mention it in each prompt (e.g., "pixar style acting and timing", "sci-fi style cinematic scene")

For EACH scene, provide:
1. A brief scene title (2-5 words describing the scene)
2. A full LTX-2 video generation prompt that includes:
   - Detailed scene description with lighting, atmosphere, and setting
   - Specific camera movements (handheld tracking, crane up, dolly back, zoom in/out, pan, etc.)
   - Character descriptions with age, appearance, clothing when relevant
   - Dialogue with character attribution and tone (e.g., "Character (whispering): 'dialogue'")
   - Action sequences with timing and beats ("A beat.", "then", "suddenly")
   - Cinematography details (depth of field, bokeh, motion blur, atmospheric effects)
   - Style mention if a style was specified
   - Write 4-12 sentences in a flowing narrative style, present tense

The prompts should match the detailed, professional style of LTX-2 examples with rich cinematography descriptions, specific camera work, and natural dialogue integration.

Respond ONLY with valid JSON in the following format (no markdown, no explanation):
{{
    "scenes": [
        {{
            "title": "Scene 1 title here",
            "prompt": "Your full LTX-2 scene prompt for scene 1 here"
        }},
        {{
            "title": "Scene 2 title here",
            "prompt": "Your full LTX-2 scene prompt for scene 2 here"
        }},
        ...
        {{
            "title": "Scene {num_scenes} title here",
            "prompt": "Your full LTX-2 scene prompt for scene {num_scenes} here"
        }}
    ]
}}

Make sure to generate exactly {num_scenes} scenes in the array."""
    
    try:
        print("🔄 Making API call to generate all scenes...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.9,  # Higher creativity
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract scenes from response
        if "scenes" in result and isinstance(result["scenes"], list):
            prompts = []
            for i, scene in enumerate(result["scenes"], 1):
                if "title" in scene and "prompt" in scene:
                    prompts.append({
                        "title": scene["title"],
                        "prompt": scene["prompt"]
                    })
                    print(f"✓ Generated Scene {i}/{num_scenes}: {scene['title']}")
                else:
                    print(f"⚠ Warning: Scene {i} is missing 'title' or 'prompt' field, skipping...")
            
            if len(prompts) != num_scenes:
                print(f"⚠ Warning: Expected {num_scenes} scenes, but received {len(prompts)} scenes.")
            
            return prompts
        else:
            print("❌ Error: Response did not contain 'scenes' array in expected format.")
            return []
            
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ Error generating scenes: {str(e)}")
        return []


def save_prompts_to_json(prompts: List[Dict[str, str]], filename: str = "ltx2_prompts.json"):
    """Save generated prompts to JSON file."""
    output = {
        "prompts": prompts
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(prompts)} prompts to {filename}")


def main():
    """Main function to generate and save LTX-2 prompts as a short film script."""
    parser = argparse.ArgumentParser(
        description="Generate LTX-2 video generation prompts as a short film script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_ltx2_prompts.py --title "The Lost Key" --description "A detective searches for a missing key" --style "film noir" --scenes 5
  python generate_ltx2_prompts.py -t "Space Adventure" -d "An astronaut discovers an alien artifact" -s "cyberpunk" -n 7 -o my_script.json
  python generate_ltx2_prompts.py  # Interactive mode (will prompt for inputs)

Style options:
  Animation: stop-motion, 2D/3D animation, claymation, hand-drawn, Studio Ghibli style
  Stylized: comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated, Studio Ghibli aesthetic
  Cinematic: period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary
        """
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        help="Story title for the short film"
    )
    parser.add_argument(
        "-d", "--description",
        type=str,
        help="Story description to provide context for generation"
    )
    parser.add_argument(
        "-s", "--style",
        type=str,
        help="Visual/cinematic style (e.g., 'film noir', 'cyberpunk', 'stop-motion', 'fantasy')"
    )
    parser.add_argument(
        "-n", "--scenes",
        type=int,
        default=5,
        help="Number of scenes in the script (default: 5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filename (default: {title}_script.json)"
    )
    
    args = parser.parse_args()
    
    print("🎬 LTX-2 Short Film Script Generator")
    print("=" * 50)
    
    # Get story title
    if args.title:
        story_title = args.title.strip()
    else:
        story_title = input("Enter the story title for your short film: ").strip()
        if not story_title:
            story_title = "Untitled Short Film"
            print(f"Using default title: {story_title}")
    
    # Get story description
    if args.description:
        story_description = args.description.strip()
        print(f"Story Description: {story_description}")
    else:
        story_description = input("\nEnter a description of your story (what the story is about, key themes, characters, etc.): ").strip()
        if not story_description:
            story_description = ""
            print("No story description provided - will generate based on title only")
        else:
            print(f"Story description received: {story_description[:100]}..." if len(story_description) > 100 else f"Story description received")
    
    # Get style
    if args.style is not None:
        style = args.style.strip()
        print(f"Style: {style}")
    else:
        print("\nStyle options:")
        print("  - Animation: stop-motion, 2D/3D animation, claymation, hand-drawn, Studio Ghibli style")
        print("  - Stylized: comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated, Studio Ghibli aesthetic")
        print("  - Cinematic: period drama, film noir, fantasy, epic space opera, thriller, modern romance, experimental film, arthouse, documentary")
        style = input("\nEnter the visual/cinematic style (e.g., 'film noir', 'cyberpunk', 'stop-motion', 'fantasy', 'Studio Ghibli style'): ").strip()
        if not style:
            style = ""
            print("No specific style selected - will use default cinematic style")
        else:
            print(f"Selected style: {style}")
    
    # Get number of scenes
    if args.scenes:
        num_scenes = args.scenes
        if num_scenes < 1:
            print("⚠ Warning: Number of scenes must be at least 1. Using default: 5")
            num_scenes = 5
    else:
        try:
            num_scenes = int(input("\nHow many scenes should the script have? (default: 5): ") or "5")
            if num_scenes < 1:
                num_scenes = 5
        except ValueError:
            num_scenes = 5
    
    print(f"\n📝 Generating story outline for '{story_title}'...")
    story_outline = generate_story_outline(story_title, story_description, num_scenes, style)
    
    if story_outline:
        print("\n📋 Story Outline:")
        print("-" * 50)
        print(story_outline)
        print("-" * 50)
    
    print(f"\n🎬 Generating {num_scenes} scene prompts for '{story_title}'...")
    print("This may take a moment...\n")
    
    prompts = generate_ltx2_prompts(story_title, story_description, num_scenes, story_outline, style)
    
    if prompts:
        print(f"\n✅ Successfully generated {len(prompts)} scene prompts!\n")
        
        # Display all scenes
        print("📖 Script Summary:")
        print("=" * 50)
        for i, p in enumerate(prompts, 1):
            print(f"\nScene {i}: {p['title']}")
            print(f"   {p['prompt'][:120]}...")
        
        # Save to file
        if args.output:
            output_file = args.output
        else:
            default_filename = f"{story_title.replace(' ', '_')}_script.json"
            output_file = input(f"\nEnter output filename (default: {default_filename}): ") or default_filename
        save_prompts_to_json(prompts, output_file)
        
        # Also print JSON to console
        print("\n📄 Generated JSON:")
        print("=" * 50)
        output_json = json.dumps({"prompts": prompts}, indent=2, ensure_ascii=False)
        print(output_json)
    else:
        print("❌ Failed to generate any scene prompts. Please check your API key and try again.")


if __name__ == "__main__":
    main()