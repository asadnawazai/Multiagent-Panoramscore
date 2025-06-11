import os
import openai
from dotenv import load_dotenv

class AIAnalyzer:
    def __init__(self):
        """
        Initialize the AIAnalyzer with API key and field keys
        """
        # Load API key from .env file
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("WARNING: OPENAI_API_KEY environment variable not found. AI analysis will not work.")
        else:
            print("OpenAI API key loaded successfully")
            openai.api_key = api_key
            
        self.field_keys = [
            "Mls Listing",
            "Build Year",
            "Land Use Code",
            "Flood Risk Score",
            "Zoning Record",
            "Outdated Tax Delta",
            "Infrastructure Opacity",
            "Regional Data Variation",
            "Climate Score"
        ]
    
    def analyze_text(self, extracted_text):
        """
        Analyze the extracted text using OpenAI GPT-3.5-turbo to extract structured data
        
        Args:
            extracted_text (str): The text extracted from OCR
            
        Returns:
            dict: Dictionary containing extracted structured data
        """
        try:
            # Check if we have text to analyze
            if not extracted_text or len(extracted_text.strip()) < 10:
                print(f"Extracted text too short or empty: {extracted_text}")
                return {
                    "success": False,
                    "error": "Insufficient text to analyze"
                }
                
            # Create the prompt for GPT
            prompt = self._build_prompt(extracted_text)
            print(f"Sending prompt to OpenAI: {prompt[:100]}...")
            
            # Check if API key is set
            if not openai.api_key:
                print("OpenAI API Key is not set. Check environment variables.")
                return {
                    "success": False, 
                    "error": "OpenAI API Key is not configured"
                }
            
            # Call the OpenAI API with older format (pre-1.0)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from scanned documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract and parse the response
            result = response.choices[0].message['content']
            print(f"Received response from OpenAI: {result[:100]}...")
            
            # Add the complete extracted text to the result
            result_dict = self._parse_result(result)
            result_dict["complete_text"] = extracted_text
            
            # Ensure all expected fields are present
            for key in self.field_keys:
                if key not in result_dict:
                    result_dict[key] = "not found"
            
            return {
                "success": True,
                "analyzed_data": result_dict
            }
            
        except Exception as e:
            import traceback
            print(f"Error in AI analysis: {str(e)}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_prompt(self, extracted_text):
        """
        Build the prompt for GPT based on extracted text
        
        Args:
            extracted_text (str): The text extracted from OCR
            
        Returns:
            str: The prompt for GPT
        """
        # Create a string of keys to extract
        keys_str = '\n'.join([f"- {key}" for key in self.field_keys])
        
        # Build the prompt with improved instructions for understanding document layout
        prompt = f"""I have extracted the following text from a document using OCR. The text may not be in sequential order because OCR extracts text from different parts of the document in a non-linear fashion:

{extracted_text}

Based on this text, please extract values for the following fields. Note that keys and values might appear separately in the text, and you need to match them correctly:
{keys_str}

IMPORTANT INSTRUCTIONS:
1. For each field, search the entire text for its value, even if the key and value are not adjacent in the OCR output.
2. Pay attention to formatting patterns - many values appear near their corresponding keys in the original document but may be extracted separately.
3. For example, if you see "Build Year" in one place and "2003" nearby (but not necessarily adjacent in the extracted text), associate 2003 with Build Year.
4. Look for patterns of information typically found in property reports or real estate documents.

For each field, if you find a matching value in the text, provide it. If not found, respond with "not found".
Format your response as key-value pairs, one per line:

Mls Listing: [value or "not found"]
Build Year: [value or "not found"]
...and so on for each field.

Be precise and only extract information that is explicitly mentioned in the text. Do not make assumptions for missing data."""
        
        return prompt
    
    def _parse_result(self, result):
        """
        Parse the result from GPT into a dictionary
        
        Args:
            result (str): The response from GPT
            
        Returns:
            dict: A dictionary of key-value pairs
        """
        result_dict = {}
        
        # First initialize all keys with 'not found'
        for key in self.field_keys:
            result_dict[key] = "not found"
        
        # Process the response line by line
        for line in result.strip().split('\n'):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Handle lines with colon separator
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes or brackets if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1].strip()
                    
                # Check if this key matches one of our expected keys (case insensitive)
                matched_key = None
                for field_key in self.field_keys:
                    if field_key.lower() == key.lower():
                        matched_key = field_key
                        break
                
                # If we found a match, use the original case from field_keys
                if matched_key:
                    result_dict[matched_key] = value
                # Otherwise, just use the key as-is
                else:
                    result_dict[key] = value
                
        print(f"Parsed result dictionary: {result_dict}")
        return result_dict
