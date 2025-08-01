#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سیستم مینیمال برای تست پروژه کایلوس با استفاده از فایل‌های متنی
این فایل وابستگی‌های کمتری نیاز دارد و می‌تواند سریعتر اجرا شود
"""

import os
import json
import logging
import re
import time
from pathlib import Path

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('caelus_minimal_test.log')
    ]
)
logger = logging.getLogger(__name__)


class MinimalCAELUS:
    """نسخه ساده‌شده سیستم بررسی انطباق برای تست عملکرد بدون نیاز به کتابخانه‌های پیچیده."""
    
    def __init__(self):
        """مقداردهی اولیه کلاس."""
        self.regulations = {}
        self.design_specs = {}
        
    def load_regulations(self, file_path):
        """خواندن متن قوانین از فایل"""
        logger.info(f"خواندن قوانین از فایل: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # پردازش مواد قانونی
            regulations = {}
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # یافتن عناوین ماده‌ها
                section_match = re.match(r'^ماده (\d+):\s+(.+)$', line)
                if section_match:
                    section_num = section_match.group(1)
                    section_title = section_match.group(2)
                    current_section = f"ماده {section_num}"
                    regulations[current_section] = {
                        'title': section_title,
                        'items': []
                    }
                    continue
                
                # یافتن بند‌های هر ماده
                item_match = re.match(r'^(\d+\.\d+)\s+(.+)$', line)
                if item_match and current_section:
                    item_num = item_match.group(1)
                    item_text = item_match.group(2)
                    regulations[current_section]['items'].append({
                        'id': item_num,
                        'text': item_text
                    })
            
            self.regulations = regulations
            logger.info(f"تعداد {len(regulations)} ماده قانونی استخراج شد.")
            return regulations
        
        except Exception as e:
            logger.error(f"خطا در خواندن فایل قوانین: {e}")
            return {}

    def load_design_specs(self, file_path):
        """خواندن مشخصات طراحی از فایل"""
        logger.info(f"خواندن مشخصات طراحی از فایل: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # پردازش بخش‌های طراحی
            design_specs = {}
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # یافتن عناوین بخش‌ها
                if line.endswith(':'):
                    current_section = line[:-1]  # حذف علامت دو نقطه
                    design_specs[current_section] = []
                    continue
                    
                # افزودن محتوا به بخش فعلی
                if current_section and line:
                    design_specs[current_section].append(line)
            
            # اگر هیچ بخشی پیدا نشد، تمام متن را به عنوان یک بخش در نظر بگیر
            if not design_specs:
                design_specs["کل مستند"] = text.split('\n')
                
            self.design_specs = design_specs
            logger.info(f"تعداد {len(design_specs)} بخش طراحی استخراج شد.")
            return design_specs
        
        except Exception as e:
            logger.error(f"خطا در خواندن فایل طراحی: {e}")
            return {}

    def check_compliance(self):
        """بررسی انطباق طراحی با قوانین"""
        logger.info("شروع بررسی انطباق...")
        
        if not self.regulations or not self.design_specs:
            logger.error("قوانین یا طراحی بارگذاری نشده‌اند.")
            return {}
            
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compliance_results": [],
            "summary": {
                "total_checks": 0,
                "compliant": 0,
                "non_compliant": 0,
                "needs_review": 0
            }
        }
        
        # تبدیل طراحی به یک متن برای جستجوی ساده‌تر
        design_text = "\n".join([
            f"{section}:\n" + "\n".join(content)
            for section, content in self.design_specs.items()
        ])
        
        # بررسی هر ماده قانونی
        for section_id, section_data in self.regulations.items():
            for item in section_data['items']:
                item_id = item['id']
                rule_text = item['text']
                
                # استخراج کلمات کلیدی و اعداد از متن قانون
                keywords = self._extract_keywords(rule_text)
                numbers = self._extract_numbers(rule_text)
                
                # بررسی کنیم آیا تمام کلمات کلیدی در طراحی وجود دارند
                keywords_found = all(keyword.lower() in design_text.lower() for keyword in keywords)
                
                # ارزیابی انطباق ساده بر اساس وجود کلمات کلیدی و ارزش‌های عددی
                compliance_status = "compliant"
                compliance_issues = []
                
                if not keywords_found:
                    compliance_status = "non_compliant"
                    compliance_issues.append(f"کلمات کلیدی لازم در طراحی یافت نشدند: {', '.join(keywords)}")
                
                # بررسی مقادیر عددی خاص مانند حداقل ضخامت عایق، مقاومت زلزله، و غیره
                for number_info in numbers:
                    original_value = number_info['value']
                    unit = number_info['unit']
                    comparison = number_info['comparison']
                    
                    # برخی قوانین خاص را بررسی کنیم
                    if "عایق حرارتی" in rule_text.lower() and "میلی‌متر" in unit:
                        # قانون 1.3 - ضخامت عایق حرارتی
                        if "50" in original_value and comparison == "حداقل":
                            # در طراحی باید بررسی شود آیا ضخامت عایق کافی است
                            if "عایق حرارتی کافی" not in design_text.lower() or "فاقد عایق" in design_text.lower():
                                compliance_status = "non_compliant"
                                compliance_issues.append(f"عایق حرارتی کافی با ضخامت حداقل {original_value} {unit} نیاز است")
                                
                    elif "زلزله" in rule_text.lower() and "شدت" in rule_text.lower():
                        # قانون 1.4 - مقاومت در برابر زلزله
                        if "0.35" in original_value and comparison == "حداقل":
                            # در طراحی باید مقاومت زلزله بررسی شود
                            design_earthquake = re.search(r'زلزله با شدت\s+(\d+\.\d+)g', design_text)
                            if design_earthquake:
                                design_value = float(design_earthquake.group(1))
                                if design_value < 0.35:
                                    compliance_status = "non_compliant"
                                    compliance_issues.append(f"مقاومت در برابر زلزله باید حداقل 0.35g باشد، اما {design_value}g است")
                            else:
                                compliance_status = "needs_review"
                                compliance_issues.append("اطلاعات کافی درباره مقاومت در برابر زلزله یافت نشد")
                                
                    # سایر قوانین با مقادیر عددی مشابه میتوانند اضافه شوند
                
                # افزودن نتیجه به گزارش
                results["compliance_results"].append({
                    "regulation_id": f"{section_id}.{item_id}",
                    "regulation_text": rule_text,
                    "status": compliance_status,
                    "issues": compliance_issues
                })
                
                # به‌روزرسانی خلاصه
                results["summary"]["total_checks"] += 1
                results["summary"][compliance_status] += 1
                
        logger.info(f"بررسی انطباق با {results['summary']['total_checks']} قانون انجام شد.")
        return results
                
    def _extract_keywords(self, text):
        """استخراج کلمات کلیدی از متن قانون"""
        # حذف کلمات عمومی
        stopwords = ["باید", "است", "باشد", "شود", "دارای", "برای", "کلیه", "تمام", "هر", "را", "از"]
        words = text.split()
        keywords = []
        
        for word in words:
            word = re.sub(r'[^\w\sآ-ی]', '', word)  # حذف علائم نگارشی
            if len(word) > 3 and word not in stopwords:
                keywords.append(word)
                
        return keywords[:5]  # برگرداندن 5 کلمه کلیدی اول
        
    def _extract_numbers(self, text):
        """استخراج اعداد و واحدهای آنها از متن قانون"""
        number_pattern = r'(\d+(?:\.\d+)?)\s*(میلی‌متر|متر|مکعب|کیلوگرم|درجه|ساعت|مگاپاسکال|g)'
        comparison_words = {
            "حداقل": "حداقل",
            "حداکثر": "حداکثر",
            "بیش از": "حداقل", 
            "کمتر از": "حداکثر"
        }
        
        results = []
        
        for match in re.finditer(number_pattern, text):
            value = match.group(1)
            unit = match.group(2)
            
            # تعیین نوع مقایسه (حداقل/حداکثر)
            comparison = "برابر"  # مقدار پیش‌فرض
            for word, comp_type in comparison_words.items():
                if word in text[:match.start()]:
                    comparison = comp_type
                    break
                    
            results.append({
                "value": value,
                "unit": unit,
                "comparison": comparison
            })
            
        return results
        
    def generate_report(self, results, output_path="output/compliance_report.json"):
        """تولید گزارش نهایی"""
        logger.info("تولید گزارش نهایی...")
        
        # ایجاد دایرکتوری خروجی اگر وجود ندارد
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # نوشتن نتایج در فایل JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"گزارش در {output_path} ذخیره شد")
        
        # ایجاد یک گزارش متنی ساده
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("گزارش بررسی انطباق طراحی با قوانین\n")
            f.write("===================================\n\n")
            f.write(f"تاریخ بررسی: {results['timestamp']}\n\n")
            
            f.write("خلاصه نتایج:\n")
            f.write(f"- تعداد کل بررسی‌ها: {results['summary']['total_checks']}\n")
            f.write(f"- منطبق با قوانین: {results['summary']['compliant']}\n")
            f.write(f"- عدم انطباق: {results['summary']['non_compliant']}\n")
            f.write(f"- نیازمند بررسی بیشتر: {results['summary']['needs_review']}\n\n")
            
            f.write("جزئیات:\n")
            for result in results['compliance_results']:
                f.write(f"\nقانون {result['regulation_id']}:\n")
                f.write(f"  {result['regulation_text']}\n")
                f.write(f"  وضعیت: {self._translate_status(result['status'])}\n")
                
                if result['issues']:
                    f.write("  مشکلات:\n")
                    for issue in result['issues']:
                        f.write(f"    - {issue}\n")
                        
        logger.info(f"گزارش متنی در {txt_path} ذخیره شد")
        return output_path
        
    def _translate_status(self, status):
        """ترجمه وضعیت به فارسی"""
        translations = {
            "compliant": "منطبق با قوانین",
            "non_compliant": "عدم انطباق",
            "needs_review": "نیازمند بررسی بیشتر"
        }
        return translations.get(status, status)


def main():
    """تابع اصلی برنامه"""
    logger.info("شروع سیستم تست مینیمال کایلوس")
    
    # مسیرهای فایل‌ها
    regulation_file = "data/raw_pdfs/nuclear_safety_regulation.txt"
    design_file = "data/design_specs/reactor_cooling_system.txt"
    output_path = "output/minimal_compliance_report.json"
    
    # ایجاد و اجرای سیستم
    tester = MinimalCAELUS()
    tester.load_regulations(regulation_file)
    tester.load_design_specs(design_file)
    
    # بررسی انطباق و تولید گزارش
    results = tester.check_compliance()
    report_path = tester.generate_report(results, output_path)
    
    logger.info(f"بررسی انطباق کامل شد. گزارش در {report_path} ذخیره شد")
    print(f"\nگزارش بررسی انطباق در {report_path} ذخیره شد")
    
    # نمایش خلاصه نتایج
    print("\nخلاصه نتایج:")
    print(f"- تعداد کل بررسی‌ها: {results['summary']['total_checks']}")
    print(f"- منطبق با قوانین: {results['summary']['compliant']}")
    print(f"- عدم انطباق: {results['summary']['non_compliant']}")
    print(f"- نیازمند بررسی بیشتر: {results['summary']['needs_review']}")
    

if __name__ == "__main__":
    main()