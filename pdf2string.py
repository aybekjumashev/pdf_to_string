"""
PDF to String Converter Module
Professional PDF converter for Uzbek, Russian, and English documents.
Optimized for both text-based and scanned PDFs.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import io
import logging
from typing import Optional, Dict, List, Tuple, Union
import re
import cv2
import numpy as np
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFToString:
    """
    Professional PDF to string converter optimized for Uzbek documents.
    
    Features:
    - High-quality OCR for scanned documents
    - Text extraction from native PDFs
    - Table detection and formatting
    - Multi-language support (Uzbek, Russian, English)
    - Quality assessment and recommendations
    - Debug capabilities
    """
    
    def __init__(self, 
                 ocr_language: str = 'uzb+rus+eng',
                 table_detection: bool = True,
                 image_ocr: bool = False,
                 preserve_layout: bool = True,
                 save_debug_images: bool = False,
                 output_dir: str = "output"):
        """
        Initialize PDF converter.
        
        Args:
            ocr_language: Tesseract language codes (e.g., 'uzb+rus+eng')
            table_detection: Enable table detection and formatting
            image_ocr: Enable OCR on embedded images
            preserve_layout: Try to preserve original layout
            save_debug_images: Save debug images for troubleshooting
            output_dir: Directory for output files and debug images
        """
        self.ocr_language = ocr_language
        self.table_detection = table_detection
        self.image_ocr = image_ocr
        self.preserve_layout = preserve_layout
        self.save_debug_images = save_debug_images
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info(f"Tesseract found: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"Tesseract not found: {e}")
            self.ocr_available = False
    
    def convert_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Convert PDF to string with comprehensive analysis.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing:
            - text: Extracted text
            - pages: List of page texts
            - tables: Detected tables
            - images: Image descriptions
            - metadata: PDF metadata
            - quality_score: Overall quality score (0-100)
            - recommendations: Improvement suggestions
            - processing_info: Technical details
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Starting conversion of: {pdf_path}")
        
        try:
            # Open PDF
            try:
                doc = fitz.open(str(pdf_path))
            except AttributeError:
                doc = fitz.Document(str(pdf_path))
            
            # Initialize result structure
            result = {
                'text': '',
                'pages': [],
                'tables': [],
                'images': [],
                'metadata': doc.metadata or {},
                'quality_score': 0.0,
                'recommendations': [],
                'processing_info': {
                    'total_pages': len(doc),
                    'ocr_pages': 0,
                    'direct_text_pages': 0,
                    'strategies_used': [],
                    'processing_time': 0
                }
            }
            
            logger.info(f"PDF opened successfully: {len(doc)} pages")
            
            # Process each page
            total_quality = 0.0
            page_strategies = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}/{len(doc)}")
                
                page_result = self._process_page(page, page_num, pdf_path.stem)
                
                result['pages'].append(page_result['text'])
                result['tables'].extend(page_result.get('tables', []))
                result['images'].extend(page_result.get('images', []))
                total_quality += page_result.get('quality_score', 0.0)
                
                # Track processing strategy
                if page_result.get('used_ocr', False):
                    result['processing_info']['ocr_pages'] += 1
                    page_strategies.append('ocr')
                else:
                    result['processing_info']['direct_text_pages'] += 1
                    page_strategies.append('direct')
            
            # Calculate overall metrics
            result['quality_score'] = total_quality / len(doc) if len(doc) > 0 else 0.0
            result['processing_info']['strategies_used'] = list(set(page_strategies))
            
            # Combine all page texts
            result['text'] = self._combine_pages(result['pages'])
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(result)
            
            # Log final results
            logger.info(f"Conversion completed - Quality: {result['quality_score']:.1f}/100")
            logger.info(f"Total text length: {len(result['text'])} characters")
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _process_page(self, page, page_num: int, pdf_name: str) -> Dict[str, any]:
        """Process a single PDF page using optimized strategy."""
        result = {
            'text': '',
            'tables': [],
            'images': [],
            'quality_score': 0.0,
            'used_ocr': False
        }
        
        # Try direct text extraction first
        direct_text = page.get_text()
        meaningful_chars = sum(1 for c in direct_text if c.isalnum())
        
        if meaningful_chars > 100:
            # Good direct text available
            logger.info(f"Page {page_num + 1}: Using direct text extraction ({meaningful_chars} chars)")
            result['text'] = direct_text
            result['quality_score'] = 95.0
            result['used_ocr'] = False
        else:
            # Need OCR
            logger.info(f"Page {page_num + 1}: Using OCR (direct text: {meaningful_chars} chars)")
            ocr_result = self._ocr_page_optimized(page, page_num, pdf_name)
            result['text'] = ocr_result.get('text', direct_text)
            result['quality_score'] = ocr_result.get('score', 0.0)
            result['used_ocr'] = True
        
        # Process tables if enabled
        if self.table_detection and result['text']:
            tables = self._extract_tables(page, page_num)
            result['tables'] = tables
            
            # Integrate tables into text
            for table in tables:
                result['text'] = self._integrate_table_text(result['text'], table)
        
        # Process images if enabled
        if self.image_ocr:
            images = self._extract_images(page, page_num)
            result['images'] = images
        
        # Clean and format text
        result['text'] = self._clean_text(result['text'])
        
        return result
    
    def _ocr_page_optimized(self, page, page_num: int, pdf_name: str) -> Dict[str, any]:
        """Optimized OCR using the proven strategy."""
        if not self.ocr_available:
            logger.warning("OCR not available")
            return {'text': '', 'score': 0.0}
        
        try:
            # Use the proven high-resolution strategy
            mat = fitz.Matrix(5.5, 5.5)  # High resolution that works
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save debug image if requested
            if self.save_debug_images:
                debug_path = self.output_dir / f"{pdf_name}_page_{page_num + 1}_debug.png"
                pix.save(str(debug_path))
                logger.info(f"Debug image saved: {debug_path}")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Use the proven OCR configuration
            text = pytesseract.image_to_string(
                image,
                lang=self.ocr_language,
                config='--psm 3'  # Proven configuration
            )
            
            if text.strip():
                score = self._calculate_quality_score(text)
                logger.info(f"OCR completed - Score: {score:.1f}, Length: {len(text.strip())}")
                return {'text': text, 'score': score}
            else:
                logger.warning("OCR returned empty text")
                return {'text': '', 'score': 0.0}
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {'text': '', 'score': 0.0}
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate OCR quality score optimized for Uzbek documents."""
        if not text.strip():
            return 0.0
        
        score = 0.0
        text_length = len(text)
        
        # Base score for text length
        score += min(text_length / 2000, 1.0) * 30
        
        # Character composition analysis
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        punct = sum(1 for c in text if c in '.,!?:;-()[]{}"\' ')
        
        if text_length > 0:
            letter_ratio = letters / text_length
            meaningful_ratio = (letters + digits + spaces + punct) / text_length
            
            score += letter_ratio * 40  # Reward high letter content
            score += meaningful_ratio * 20  # Reward meaningful characters
        
        # Language-specific word detection
        words = text.lower().split()
        
        # Uzbek document indicators
        uzbek_indicators = [
            'respublikasi', 'prezident', 'qarori', 'vazirlik', 'dastur',
            'uchun', 'tomonidan', 'bilan', 'haqida', 'asosan', 'maqsad',
            'natija', 'chorak', 'yil', 'son', 'ilova'
        ]
        
        # Russian words
        russian_words = [
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'о', 'что', 'как'
        ]
        
        # English words
        english_words = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'was'
        ]
        
        all_indicators = uzbek_indicators + russian_words + english_words
        
        word_matches = sum(1 for word in words if any(indicator in word for indicator in all_indicators))
        if len(words) > 0:
            word_score = (word_matches / len(words)) * 25
            score += word_score
        
        # Document structure indicators
        has_numbers = bool(re.search(r'\d{4}', text))  # Years
        has_proper_sentences = len([s for s in text.split('.') if len(s.strip()) > 10]) > 0
        
        if has_numbers:
            score += 5
        if has_proper_sentences:
            score += 10
        
        # Penalty for obvious OCR errors
        weird_chars = sum(1 for c in text if ord(c) > 1000 and c not in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        if text_length > 0:
            weird_ratio = weird_chars / text_length
            score -= weird_ratio * 30
        
        return max(min(score, 100.0), 0.0)
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        """Extract and format tables from page."""
        tables = []
        
        try:
            # Simple table detection based on text layout
            blocks = page.get_text("dict")["blocks"]
            table_candidates = self._detect_table_structures(blocks)
            
            for i, table_data in enumerate(table_candidates):
                if table_data:
                    formatted_table = self._format_table_data(table_data)
                    tables.append({
                        'page': page_num + 1,
                        'table_id': i + 1,
                        'data': table_data,
                        'formatted': formatted_table
                    })
                    
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _detect_table_structures(self, blocks) -> List[List[List[str]]]:
        """Detect table-like structures in text blocks."""
        table_candidates = []
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                lines = block.get("lines", [])
                if len(lines) > 2:
                    # Look for aligned text that might be a table
                    rows = []
                    for line in lines:
                        spans = line.get("spans", [])
                        row_text = ""
                        for span in spans:
                            row_text += span.get("text", "")
                        
                        if row_text.strip():
                            # Split by multiple spaces or tabs
                            cells = re.split(r'\s{3,}|\t+', row_text.strip())
                            if len(cells) > 1:
                                rows.append(cells)
                    
                    if len(rows) > 1:
                        table_candidates.append(rows)
        
        return table_candidates
    
    def _format_table_data(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text."""
        if not table_data:
            return ""
        
        try:
            # Try to create a pandas DataFrame for nice formatting
            max_cols = max(len(row) for row in table_data)
            
            # Pad rows to same length
            normalized_data = []
            for row in table_data:
                padded_row = row + [''] * (max_cols - len(row))
                normalized_data.append(padded_row)
            
            df = pd.DataFrame(normalized_data[1:], columns=normalized_data[0])
            return df.to_string(index=False, max_colwidth=30)
            
        except Exception:
            # Fallback to simple formatting
            formatted_rows = []
            for row in table_data:
                formatted_rows.append(" | ".join(str(cell) for cell in row))
            return "\n".join(formatted_rows)
    
    def _integrate_table_text(self, text: str, table: Dict) -> str:
        """Integrate formatted table into text."""
        table_marker = f"\n\n[TABLE {table['table_id']} - Page {table['page']}]\n"
        table_marker += table['formatted']
        table_marker += f"\n[END TABLE {table['table_id']}]\n\n"
        
        return text + table_marker
    
    def _extract_images(self, page, page_num: int) -> List[Dict]:
        """Extract and OCR images from page."""
        images = []
        
        if not self.ocr_available:
            return images
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # OCR the image
                        text = pytesseract.image_to_string(
                            image,
                            lang=self.ocr_language,
                            config='--psm 6'
                        )
                        
                        if text.strip():
                            images.append({
                                'page': page_num + 1,
                                'image_id': img_index + 1,
                                'text': text.strip(),
                                'size': (pix.width, pix.height)
                            })
                    
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_index}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        
        return images
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        text = re.sub(r'([.,!?:;])\s*([А-Яа-яË])', r'\1 \2', text)
        
        # Remove standalone single characters (OCR artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) <= 1:
                continue
            
            # Remove obvious OCR artifacts
            if len(line) > 0:
                alpha_ratio = sum(1 for c in line if c.isalpha()) / len(line)
                if alpha_ratio < 0.3 and len(line) < 5:
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _combine_pages(self, pages: List[str]) -> str:
        """Combine page texts intelligently."""
        if not pages:
            return ""
        
        # Join pages with double newlines
        combined = '\n\n'.join(page for page in pages if page.strip())
        
        # Final cleanup
        combined = re.sub(r'\n\s*\n\s*\n+', '\n\n', combined)
        
        return combined.strip()
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations based on processing results."""
        recommendations = []
        quality = result['quality_score']
        
        if quality >= 80:
            recommendations.append("Excellent OCR quality! The text should be highly accurate.")
        elif quality >= 60:
            recommendations.append("Good OCR quality. Minor manual corrections may be needed.")
        elif quality >= 40:
            recommendations.append("Moderate OCR quality. Please review and correct errors.")
        elif quality >= 20:
            recommendations.append("Poor OCR quality. Significant manual correction required.")
        else:
            recommendations.append("Very poor OCR quality. Consider:")
            recommendations.append("  - Rescanning at higher resolution (600+ DPI)")
            recommendations.append("  - Using professional OCR software")
            recommendations.append("  - Manual retyping for critical documents")
        
        # Processing-specific recommendations
        info = result['processing_info']
        if info['ocr_pages'] > info['direct_text_pages']:
            recommendations.append("Document is mostly scanned - OCR quality depends on scan quality")
        
        if len(result['text']) < 500:
            recommendations.append("Very little text extracted - check if document contains text")
        
        if result['tables']:
            recommendations.append(f"Found {len(result['tables'])} tables - verify table formatting")
        
        return recommendations
    
    def save_result(self, result: Dict, output_path: Union[str, Path] = None) -> Path:
        """Save conversion result to file."""
        if output_path is None:
            output_path = self.output_dir / "conversion_result.txt"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("PDF TO STRING CONVERSION RESULT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("SUMMARY:\n")
            f.write(f"Quality Score: {result['quality_score']:.1f}/100\n")
            f.write(f"Total Pages: {result['processing_info']['total_pages']}\n")
            f.write(f"OCR Pages: {result['processing_info']['ocr_pages']}\n")
            f.write(f"Direct Text Pages: {result['processing_info']['direct_text_pages']}\n")
            f.write(f"Text Length: {len(result['text'])} characters\n")
            f.write(f"Tables Found: {len(result['tables'])}\n")
            f.write(f"Images Processed: {len(result['images'])}\n\n")
            
            # Recommendations
            if result['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in result['recommendations']:
                    f.write(f"• {rec}\n")
                f.write("\n")
            
            # Metadata
            if result['metadata']:
                f.write("DOCUMENT METADATA:\n")
                for key, value in result['metadata'].items():
                    if value:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Main text
            f.write("EXTRACTED TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(result['text'])
            f.write("\n\n")
            
            # Tables
            if result['tables']:
                f.write("DETECTED TABLES:\n")
                f.write("-" * 40 + "\n")
                for table in result['tables']:
                    f.write(f"\nTable {table['table_id']} (Page {table['page']}):\n")
                    f.write(table['formatted'])
                    f.write("\n")
            
            # Images
            if result['images']:
                f.write("\nIMAGE OCR RESULTS:\n")
                f.write("-" * 40 + "\n")
                for img in result['images']:
                    f.write(f"\nImage {img['image_id']} (Page {img['page']}):\n")
                    f.write(f"Size: {img['size']}\n")
                    f.write(f"Text: {img['text']}\n")
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def process_multiple_pdfs(self, pdf_directory: Union[str, Path], 
                             pattern: str = "*.pdf") -> Dict[str, Dict]:
        """Process multiple PDF files in a directory."""
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir} matching {pattern}")
            return {}
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        results = {}
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                result = self.convert_pdf(pdf_file)
                results[pdf_file.name] = result
                
                # Save individual result
                output_file = self.output_dir / f"{pdf_file.stem}_result.txt"
                self.save_result(result, output_file)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results[pdf_file.name] = {'error': str(e)}
        
        # Save summary
        self._save_batch_summary(results)
        
        return results
    
    def _save_batch_summary(self, results: Dict[str, Dict]):
        """Save summary of batch processing."""
        summary_path = self.output_dir / "batch_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            successful = sum(1 for r in results.values() if 'error' not in r)
            failed = len(results) - successful
            
            f.write(f"Total files: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            
            for filename, result in results.items():
                if 'error' in result:
                    f.write(f"❌ {filename}: {result['error']}\n")
                else:
                    quality = result.get('quality_score', 0)
                    f.write(f"✅ {filename}: Quality {quality:.1f}/100\n")
        
        logger.info(f"Batch summary saved to: {summary_path}")


# Convenience functions
def pdf_to_string(pdf_path: Union[str, Path], **kwargs) -> str:
    """
    Simple function to convert PDF to string.
    
    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional options for PDFToString
    
    Returns:
        Extracted text as string
    """
    converter = PDFToString(**kwargs)
    result = converter.convert_pdf(pdf_path)
    return result['text']


def pdf_to_dict(pdf_path: Union[str, Path], **kwargs) -> Dict:
    """
    Convert PDF to dictionary with full details.
    
    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional options for PDFToString
    
    Returns:
        Dictionary with text, tables, images, and metadata
    """
    converter = PDFToString(**kwargs)
    return converter.convert_pdf(pdf_path)


def batch_convert_pdfs(pdf_directory: Union[str, Path], 
                      output_directory: Union[str, Path] = "output",
                      **kwargs) -> Dict[str, Dict]:
    """
    Convert multiple PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files
        **kwargs: Additional options for PDFToString
    
    Returns:
        Dictionary mapping filenames to conversion results
    """
    kwargs['output_dir'] = output_directory
    converter = PDFToString(**kwargs)
    return converter.process_multiple_pdfs(pdf_directory)


# CLI interface
def main():
    """Command line interface for PDF conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDF files to text")
    parser.add_argument("pdf_path", help="Path to PDF file or directory")
    parser.add_argument("-o", "--output", help="Output directory", default="output")
    parser.add_argument("-l", "--language", help="OCR language", default="uzb+rus+eng")
    parser.add_argument("--no-tables", action="store_true", help="Disable table detection")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--batch", action="store_true", help="Process directory of PDFs")
    
    args = parser.parse_args()
    
    # Setup converter
    converter = PDFToString(
        ocr_language=args.language,
        table_detection=not args.no_tables,
        save_debug_images=args.debug,
        output_dir=args.output
    )
    
    pdf_path = Path(args.pdf_path)
    
    if args.batch or pdf_path.is_dir():
        # Batch processing
        print(f"Processing PDFs in directory: {pdf_path}")
        results = converter.process_multiple_pdfs(pdf_path)
        print(f"Processed {len(results)} files. Check {args.output} for results.")
    else:
        # Single file processing
        print(f"Processing PDF: {pdf_path}")
        result = converter.convert_pdf(pdf_path)
        
        output_file = converter.save_result(result)
        
        print(f"Conversion completed!")
        print(f"Quality Score: {result['quality_score']:.1f}/100")
        print(f"Text Length: {len(result['text'])} characters")
        print(f"Results saved to: {output_file}")
        
        # Show preview
        if result['text']:
            print(f"\nPreview (first 500 characters):")
            print("-" * 50)
            print(result['text'][:500])
            if len(result['text']) > 500:
                print("...")


if __name__ == "__main__":
    main()