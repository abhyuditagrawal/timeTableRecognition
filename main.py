import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class LectureInfo:
    """Data class to store lecture information"""
    day: str
    time_slots: List[str]
    subject: str
    faculty: str
    classroom: str
    batch: str
    lecture_type: str

class RobustTimetableAnalyzer:
    def __init__(self):
        self.time_slots = [
            "8:00-9:00", "9:00-10:00", "10:00-11:00", "11:00-12:00",
            "12:00-1:00", "1:00-2:00", "2:00-3:00", "3:00-4:00", "4:00-5:00", "5:00-6:00"
        ]
        
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday"]
        self.subject_codes = ["DSA", "DM", "DLD", "PEM", "P&S", "CN"]
        self.special_entries = ["BREAK", "LIBRARY"]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhanced preprocessing"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        return thresh

    def extract_with_regions(self, image_path: str) -> Dict:
        """Extract text using predefined regions instead of line detection"""
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        print(f"Image dimensions: {w} x {h}")
        
        # Define approximate regions based on typical timetable layout
        # Skip the header row and time column
        
        # Calculate approximate column and row sizes
        time_col_width = w // 6  # First column for time (roughly 1/6 of width)
        day_col_width = (w - time_col_width) // 4  # Remaining width divided by 4 days
        header_height = h // 12  # Header row height
        row_height = (h - header_height) // len(self.time_slots)  # Remaining height divided by time slots
        
        print(f"Calculated dimensions - Day column width: {day_col_width}, Row height: {row_height}")
        
        timetable_data = {
            "schedule": {},
            "raw_lectures": [],
            "debug_info": {
                "regions_processed": 0,
                "regions_with_content": 0,
                "extracted_texts": []
            }
        }
        
        # Initialize schedule
        for day in self.days:
            timetable_data["schedule"][day] = {}
            for time_slot in self.time_slots:
                timetable_data["schedule"][day][time_slot] = None
        
        # Extract text from each day/time combination
        for day_idx, day in enumerate(self.days):
            for time_idx, time_slot in enumerate(self.time_slots):
                
                # Calculate region coordinates
                x1 = time_col_width + (day_idx * day_col_width) + 5  # Add small padding
                x2 = time_col_width + ((day_idx + 1) * day_col_width) - 5
                y1 = header_height + (time_idx * row_height) + 5
                y2 = header_height + ((time_idx + 1) * row_height) - 5
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w-1))
                x2 = max(x1+1, min(x2, w))
                y1 = max(0, min(y1, h-1))
                y2 = max(y1+1, min(y2, h))
                
                timetable_data["debug_info"]["regions_processed"] += 1
                
                # Extract region
                region = img[y1:y2, x1:x2]
                
                if region.size == 0:
                    continue
                
                # Extract text from region
                text = self.extract_text_from_region(region)
                
                timetable_data["debug_info"]["extracted_texts"].append({
                    "day": day,
                    "time_slot": time_slot,
                    "region": (x1, y1, x2, y2),
                    "text": text,
                    "region_size": (x2-x1, y2-y1)
                })
                
                if text and len(text.strip()) > 2:
                    timetable_data["debug_info"]["regions_with_content"] += 1
                    
                    # Parse the extracted text
                    parsed_content = self.parse_cell_content(text)
                    
                    if parsed_content:
                        # Add to schedule
                        timetable_data["schedule"][day][time_slot] = {
                            "subject": parsed_content['subject'],
                            "faculty": parsed_content['faculty'],
                            "classroom": parsed_content['classroom'],
                            "batch": parsed_content['batch'],
                            "type": parsed_content['lecture_type']
                        }
                        
                        # Add to raw lectures
                        lecture = LectureInfo(
                            day=day,
                            time_slots=[time_slot],
                            subject=parsed_content['subject'],
                            faculty=parsed_content['faculty'],
                            classroom=parsed_content['classroom'],
                            batch=parsed_content['batch'],
                            lecture_type=parsed_content['lecture_type']
                        )
                        
                        timetable_data["raw_lectures"].append(asdict(lecture))
        
        return timetable_data

    def extract_text_from_region(self, region: np.ndarray) -> str:
        """Extract text from a specific image region"""
        if region.size == 0:
            return ""
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region
        
        # Resize region if it's too small (OCR works better on larger text)
        height, width = gray_region.shape
        if height < 100 or width < 100:
            scale_factor = max(2, 150 // min(height, width))
            new_width = width * scale_factor
            new_height = height * scale_factor
            gray_region = cv2.resize(gray_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # OCR with multiple configurations
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 7',
            r'--oem 3 --psm 13'
        ]
        
        best_text = ""
        max_length = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(cleaned, config=config).strip()
                if len(text) > max_length:
                    max_length = len(text)
                    best_text = text
            except:
                continue
        
        return best_text

    def parse_cell_content(self, text: str) -> Dict:
        """Parse cell content: Subject, [Faculty], Batch, Location"""
        if not text or len(text.strip()) < 2:
            return None
        
        # Clean and split text
        lines = [line.strip() for line in text.replace('\n', ' ').split() if line.strip()]
        full_text = ' '.join(lines)
        
        print(f"Parsing text: '{full_text}'")  # Debug output
        
        result = {
            'subject': '',
            'faculty': '',
            'batch': '',
            'classroom': '',
            'lecture_type': '',
            'raw_text': text
        }
        
        # Handle special cases
        text_upper = full_text.upper()
        if 'BREAK' in text_upper:
            result.update({
                'subject': 'BREAK',
                'lecture_type': 'BREAK'
            })
            return result
        elif 'LIBRARY' in text_upper:
            result.update({
                'subject': 'LIBRARY', 
                'lecture_type': 'LIBRARY'
            })
            return result
        
        # Extract subject
        for subject_code in self.subject_codes:
            if subject_code in full_text:
                result['subject'] = subject_code
                break
        
        # Extract faculty (in square brackets)
        faculty_patterns = [
            r'\[([A-Z]{2,4})\]',
            r'\[([A-Za-z]{2,4})\]'
        ]
        
        for pattern in faculty_patterns:
            match = re.search(pattern, full_text)
            if match:
                result['faculty'] = match.group(1).upper()
                break
        
        # Extract batch
        batch_match = re.search(r'\b(B[12])\b', full_text)
        if batch_match:
            result['batch'] = batch_match.group(1)
        
        # Extract classroom
        classroom_patterns = [
            r'(CR-\d+)',
            r'(TR-\d+)', 
            r'(CL-\d+)',
            r'(CC-\d+)',
            r'(CL-[A-Z]+ Lab)',
            r'(CL-[A-Za-z]+ Lab)',
            r'(LAB L\d+-IT)',
            r'(Lab L\d+-IT)',
        ]
        
        for pattern in classroom_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                result['classroom'] = match.group(1)
                break
        
        # Determine lecture type
        if 'LAB' in text_upper or 'Lab' in full_text:
            result['lecture_type'] = 'LAB'
        elif result['subject'] in self.subject_codes:
            result['lecture_type'] = 'LECTURE'
        else:
            result['lecture_type'] = 'OTHER'
        
        # Return result if we found meaningful content
        if result['subject'] or result['lecture_type'] in ['BREAK', 'LIBRARY']:
            return result
        
        return None

    def analyze_timetable(self, image_path: str) -> Dict:
        """Main analysis function"""
        print("Starting robust timetable analysis...")
        
        # Extract using region-based approach
        timetable_data = self.extract_with_regions(image_path)
        
        # Add metadata
        lectures = timetable_data["raw_lectures"]
        timetable_data["metadata"] = {
            "total_lectures": len([l for l in lectures if l['lecture_type'] == 'LECTURE']),
            "total_labs": len([l for l in lectures if l['lecture_type'] == 'LAB']),
            "total_breaks": len([l for l in lectures if l['lecture_type'] == 'BREAK']),
            "total_entries": len(lectures),
            "extraction_method": "Region-based extraction with enhanced OCR"
        }
        
        print(f"Processed {timetable_data['debug_info']['regions_processed']} regions")
        print(f"Found content in {timetable_data['debug_info']['regions_with_content']} regions")
        print(f"Extracted {len(lectures)} valid lectures")
        
        return timetable_data

    def save_results(self, timetable_data: Dict, output_file: str = "robust_timetable.json"):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(timetable_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    def print_debug_info(self, timetable_data: Dict):
        """Print debug information"""
        print("\n" + "="*60)
        print("DEBUG INFORMATION")
        print("="*60)
        
        debug_info = timetable_data.get("debug_info", {})
        
        for item in debug_info.get("extracted_texts", [])[:10]:  # Show first 10 for brevity
            print(f"\n{item['day']} {item['time_slot']}:")
            print(f"  Region: {item['region']}")
            print(f"  Size: {item['region_size']}")
            print(f"  Text: '{item['text']}'")

    def print_readable_schedule(self, timetable_data: Dict):
        """Print schedule in readable format"""
        print("\n" + "="*60)
        print("EXTRACTED TIMETABLE")
        print("="*60)
        
        for day in self.days:
            print(f"\n{day.upper()}:")
            print("-" * 40)
            
            day_schedule = timetable_data["schedule"][day]
            has_classes = False
            
            for time_slot in self.time_slots:
                entry = day_schedule[time_slot]
                if entry and entry.get('subject'):
                    has_classes = True
                    print(f"{time_slot}: {entry['subject']} [{entry['faculty']}] {entry['classroom']} {entry['batch']} ({entry['type']})")
            
            if not has_classes:
                print("No classes scheduled")

# Example usage
if __name__ == "__main__":
    analyzer = RobustTimetableAnalyzer()
    image_path = "test4.jpg"
    
    try:
        # Analyze timetable
        timetable_data = analyzer.analyze_timetable(image_path)
        
        # Print debug information
        analyzer.print_debug_info(timetable_data)
        
        # Print readable schedule
        analyzer.print_readable_schedule(timetable_data)
        
        # Print metadata
        metadata = timetable_data["metadata"]
        print(f"\nANALYSIS SUMMARY:")
        print(f"Total Lectures: {metadata['total_lectures']}")
        print(f"Total Labs: {metadata['total_labs']}")
        print(f"Total Breaks: {metadata['total_breaks']}")
        print(f"Total Entries: {metadata['total_entries']}")
        
        # Save results
        analyzer.save_results(timetable_data, "robust_timetable.json")
        
        # Save clean version without debug info
        clean_data = {k: v for k, v in timetable_data.items() if k != 'debug_info'}
        analyzer.save_results(clean_data, "clean_robust_timetable.json")
        
    except FileNotFoundError:
        print("Error: test4.jpg not found in current directory")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()