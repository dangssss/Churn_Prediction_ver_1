# ops/data_transformations.py
"""
Data transformation rules cho 4 bảng mới:
- Encoding mappings (customer_code, item_code encryption)
- Data cleaning (complaint_content regex, datetime normalization)
- Field filtering & casting
"""

import re
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib


class CustomerEncryption:
    """
    Mapping ánh xạ 1:1 mã khách hàng & mã bưu gửi.
    Simple implementation: hash MD5 (hoặc bạn có thể dùng fernet/AES nếu muốn reversible)
    """

    def __init__(self, salt: str = "ds_churn_secret_salt"):
        self.salt = salt
        self._cache: Dict[str, str] = {}

    def encode(self, code: str) -> str:
        """
        Mã hóa 1 mã khách hàng/bưu gửi.
        Dùng cache để đảm bảo ánh xạ 1:1.
        """
        if not code or not isinstance(code, str):
            return None

        code = code.strip().upper()
        if not code:
            return None

        if code in self._cache:
            return self._cache[code]

        # Hash MD5 + salt
        hasher = hashlib.md5()
        hasher.update((code + self.salt).encode("utf-8"))
        encoded = "ENC_" + hasher.hexdigest()[:16].upper()

        self._cache[code] = encoded
        return encoded

    def save_mapping(self, filepath: str) -> None:
        """Lưu mapping vào file để có thể dùng lại sau."""
        import json

        with open(filepath, "w") as f:
            json.dump(self._cache, f, indent=2)

    def load_mapping(self, filepath: str) -> None:
        """Tải mapping từ file."""
        import json

        with open(filepath, "r") as f:
            self._cache = json.load(f)


class ComplaintContentCleaner:
    """
    Xóa chuỗi 8 chữ số liên tiếp và các ký tự whitespace gây hiểu nhầm.
    """

    # Pattern: 8 chữ số liên tiếp hoặc \n, \r\n, \r, U+2028, U+2029
    PATTERN = re.compile(
        r"\d{8}"  # 8 consecutive digits
        r"|\\n"  # escaped newline
        r"|\\r\\n"  # escaped crlf
        r"|\\r"  # escaped cr
        r"|[\r\n\u2028\u2029]"  # actual whitespace chars
        r"|\\\\"  # escaped backslash
    )

    @staticmethod
    def clean(content: str, replace_with: str = " ") -> str:
        """
        Làm sạch complaint_content.
        """
        if not content or not isinstance(content, str):
            return None

        # Replace nguy hiểm patterns
        cleaned = ComplaintContentCleaner.PATTERN.sub(replace_with, content)

        # Normalize spaces
        cleaned = " ".join(cleaned.split())

        return cleaned if cleaned else None


class DatetimeNormalizer:
    """
    Chuẩn hóa datetime về format YYYY-MM-DD HH:MM:SS (múi giờ +7).
    """

    FORMATS = [
        # Microseconds first
        "%Y-%m-%d %H:%M:%S.%f",  # 2025-11-20 11:40:30.209324
        
        # 2-digit year formats (PRIORITY - must be before 4-digit to avoid conflicts)
        "%y-%m-%d %H:%M:%S",  # 20-01-01 00:00:00 → 2020-01-01 00:00:00
        "%y-%m-%d",  # 20-01-01 → 2020-01-01
        "%y/%m/%d %H:%M:%S",
        "%y/%m/%d",
        "%d/%m/%y %H:%M:%S",  # 01/01/20 12:30:00
        "%d/%m/%y",  # 01/01/20
        "%d-%m-%y %H:%M:%S",  # 01-01-20 12:30:00
        "%d-%m-%y",  # 01-01-20
        
        # 4-digit year formats
        "%Y-%m-%d %H:%M:%S",  # 2025-01-01 12:30:00
        "%Y-%m-%d",  # 2025-01-01
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",  # Vietnamese 01/01/2025 12:30:00
        "%d/%m/%Y",  # Vietnamese 01/01/2025
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
    ]

    @staticmethod
    def normalize(dt_str: str, timezone_offset_hours: int = 7) -> Optional[str]:
        """
        Cố gắng parse datetime string và trả về chuẩn format.
        Nếu parse thất bại -> return None
        """
        if not dt_str or not isinstance(dt_str, str):
            return None

        dt_str = dt_str.strip()
        if not dt_str or dt_str.lower() in ("null", "nan", ""):
            return None

        for fmt in DatetimeNormalizer.FORMATS:
            try:
                dt = datetime.strptime(dt_str, fmt)
                # Giả định timezone +7 (Vietnam)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

        # Nếu không parse được -> None
        return None


# ============================================================
# Safe Type Casting - dùng cho implicit TEXT → INT/DECIMAL/DATE
# ============================================================


class SafeTypeCaster:
    """
    Safely cast TEXT values từ CSV sang các type chuẩn.
    Trả về:
    - None nếu invalid (không 0 hoặc "" để Postgres insert NULL)
    - String chuẩn (để Postgres tự cast khi insert)
    """

    @staticmethod
    def to_int(val: Any) -> Optional[str]:
        """
        Safely cast to INT:
        - Strip whitespace
        - Handle boolean: t/f, true/false, T/F → 1/0
        - Remove non-digit chars (except -)
        - Return None if invalid
        """
        if val is None or str(val).strip() == "":
            return None
        
        s = str(val).strip().upper()
        if s in ("NULL", "NAN", "N/A", "NONE", ""):
            return None
        
        # Handle boolean strings (Postgres exports as t/f)
        if s in ("T", "TRUE", "Y", "YES", "1"):
            return "1"
        if s in ("F", "FALSE", "N", "NO", "0"):
            return "0"
        
        # Remove thousand separators & whitespace, keep only digits and -
        try:
            s_clean = re.sub(r"[^\d\-]", "", s)
            if not s_clean or s_clean == "-":
                return None
            int_val = int(s_clean)
            return str(int_val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def to_decimal(val: Any, precision: int = 3) -> Optional[str]:
        """
        Safely cast to DECIMAL:
        - Strip whitespace
        - Replace comma with dot (European format)
        - Return None if invalid
        - Keep as string để Postgres cast
        """
        if val is None or str(val).strip() == "":
            return None
        
        s = str(val).strip().upper()
        if s in ("NULL", "NAN", "N/A", ""):
            return None
        
        try:
            # Replace comma (European) with dot
            s_clean = s.replace(",", ".")
            # Remove other non-numeric chars (except .)
            s_clean = re.sub(r"[^\d\.\-]", "", s_clean)
            
            float_val = float(s_clean)
            # Format với precision
            formatted = f"{float_val:.{precision}f}"
            return formatted
        except (ValueError, TypeError):
            return None

    @staticmethod
    def to_timestamp(val: Any) -> Optional[str]:
        """
        Safely cast to TIMESTAMP:
        - Dùng DatetimeNormalizer
        - Return None if invalid
        """
        if val is None or str(val).strip() == "":
            return None
        
        val_str = str(val)
        normalized = DatetimeNormalizer.normalize(val_str)
        
        # DEBUG: Log if failed to parse
        if normalized is None and val_str.strip() not in ("", "null", "NULL", "nan", "NAN"):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse datetime: '{val_str}' - returning None")
        
        return normalized

    @staticmethod
    def clean_string(val: Any) -> Optional[str]:
        """
        Safely clean VARCHAR/TEXT:
        - Strip whitespace
        - Replace NULL-like strings (including 'None' text)
        - Return None if empty
        """
        if val is None:
            return None
        
        s = str(val).strip()
        if s == "" or s.upper() in ("NULL", "NAN", "N/A", "NONE", "#N/A"):
            return None
        
        return s


# ============================================================
# Apply transformation cho toàn bộ row
# ============================================================


def transform_bccp_orderitem_row(
    raw_row: Dict[str, Any], encrypto: Optional[CustomerEncryption] = None
) -> Optional[Dict[str, Any]]:
    """
    Transform 1 row từ bccp_orderitem CSV.
    Dùng SafeTypeCaster để cast TEXT → INT/DECIMAL/TIMESTAMP an toàn.

    CSV header:
      crm_code_enc;cms_code_enc;item_code_enc;service_code;weight_kg;
      length_size;width_size;height_size;total_fee;is_domestic;country_code;
      send_province_code;send_district_code;send_commune_code;
      rec_province_code;rec_district_code;rec_commune_code;region;
      sending_time;ending_time;rec_success;refunded;no_accepted;lost_order;
      delay_day;done;total_complaint;
      complaint114;complaint115;complaint116;complaint134;complaint194;
      complaint554;complaint595;complaint314;complaint594;complaint274;
      complaint614;complaint654;complaint234;complaint174;order_score;
      bccp_update_date
      
    Lưu ý: CSV có item_code_enc, transform sẽ map sang item_code (canonical)
    """
    if not raw_row:
        return None

    caster = SafeTypeCaster
    
    return {
        # VARCHAR(100)
        "crm_code_enc": caster.clean_string(raw_row.get("crm_code_enc")),
        "cms_code_enc": caster.clean_string(raw_row.get("cms_code_enc")),
        "item_code": caster.clean_string(raw_row.get("item_code_enc") or raw_row.get("item_code")),  # Normalized key = item_code (từ item_code_enc)
        
        # VARCHAR
        "service_code": caster.clean_string(raw_row.get("service_code")),
        "country_code": caster.clean_string(raw_row.get("country_code")),
        "region": caster.clean_string(raw_row.get("region")),
        
        # DECIMAL(10,3)
        "weight_kg": caster.to_decimal(raw_row.get("weight_kg"), 3),
        "order_score": caster.to_decimal(raw_row.get("order_score"), 3),
        
        # INT (sizes)
        "length_size": caster.to_int(raw_row.get("length_size")),
        "width_size": caster.to_int(raw_row.get("width_size")),
        "height_size": caster.to_int(raw_row.get("height_size")),
        
        # INT (fees & counts)
        "total_fee": caster.to_int(raw_row.get("total_fee")),
        "is_domestic": caster.to_int(raw_row.get("is_domestic")),
        "rec_success": caster.to_int(raw_row.get("rec_success")),
        "refunded": caster.to_int(raw_row.get("refunded")),
        "no_accepted": caster.to_int(raw_row.get("no_accepted")),
        "lost_order": caster.to_int(raw_row.get("lost_order")),
        "delay_day": caster.to_int(raw_row.get("delay_day")),
        "done": caster.to_int(raw_row.get("done")),
        "total_complaint": caster.to_int(raw_row.get("total_complaint")),
        
        # INT (province/district/commune codes)
        "send_province_code": caster.to_int(raw_row.get("send_province_code")),
        "send_district_code": caster.to_int(raw_row.get("send_district_code")),
        "send_commune_code": caster.to_int(raw_row.get("send_commune_code")),
        "rec_province_code": caster.to_int(raw_row.get("rec_province_code")),
        "rec_district_code": caster.to_int(raw_row.get("rec_district_code")),
        "rec_commune_code": caster.to_int(raw_row.get("rec_commune_code")),
        
        # INT (complaint counts)
        "complaint114": caster.to_int(raw_row.get("complaint114")),
        "complaint115": caster.to_int(raw_row.get("complaint115")),
        "complaint116": caster.to_int(raw_row.get("complaint116")),
        "complaint134": caster.to_int(raw_row.get("complaint134")),
        "complaint194": caster.to_int(raw_row.get("complaint194")),
        "complaint554": caster.to_int(raw_row.get("complaint554")),
        "complaint595": caster.to_int(raw_row.get("complaint595")),
        "complaint314": caster.to_int(raw_row.get("complaint314")),
        "complaint594": caster.to_int(raw_row.get("complaint594")),
        "complaint274": caster.to_int(raw_row.get("complaint274")),
        "complaint614": caster.to_int(raw_row.get("complaint614")),
        "complaint654": caster.to_int(raw_row.get("complaint654")),
        "complaint234": caster.to_int(raw_row.get("complaint234")),
        "complaint174": caster.to_int(raw_row.get("complaint174")),
        
        # TIMESTAMPTZ
        "sending_time": caster.to_timestamp(raw_row.get("sending_time")),
        "ending_time": caster.to_timestamp(raw_row.get("ending_time")),
        "bccp_update_date": caster.to_timestamp(raw_row.get("bccp_update_date")),
    }

def transform_cas_customer_row(
    raw_row: Dict[str, Any], encrypto: Optional[CustomerEncryption] = None
) -> Optional[Dict[str, Any]]:
    """
    Transform 1 row từ cas_customer CSV.
    Dùng SafeTypeCaster để cast TEXT → INT/DECIMAL/TIMESTAMP an toàn.

    CSV header:
      cms_code_enc;report_month;item_count;weight_kg;total_fee;
      intra_province;international;
      ser_c;ser_e;ser_m;ser_p;ser_r;ser_u;ser_l;ser_q;
      delay_day;delay_count;nodone;refunded;noaccepted;lost_order;lastday;noservice;
      dev_item;order_score;satisfaction_score;total_complaint;
      complaint114;complaint115;complaint116;complaint134;complaint194;complaint554;
      complaint595;complaint314;complaint594;complaint274;complaint614;complaint654;
      complaint234;complaint174;updated_at
    """
    if not raw_row:
        return None

    caster = SafeTypeCaster
    
    # Parse report_month to DATE format (YYYY-MM-DD)
    report_month_raw = raw_row.get("report_month")
    report_month_date = None
    if report_month_raw:
        # Try to parse as date and format as YYYY-MM-DD for PostgreSQL DATE type
        parsed_date = DatetimeNormalizer.normalize(str(report_month_raw))
        if parsed_date:
            # Extract just the date part (YYYY-MM-DD)
            report_month_date = parsed_date.split(' ')[0]
    
    return {
        # VARCHAR(100)
        "cms_code_enc": caster.clean_string(raw_row.get("cms_code_enc")),
        # DATE
        "report_month": report_month_date,
        
        # BIGINT
        "item_count": caster.to_int(raw_row.get("item_count")),
        "total_fee": caster.to_int(raw_row.get("total_fee")),
        
        # DECIMAL(12,3)
        "weight_kg": caster.to_decimal(raw_row.get("weight_kg"), 3),
        
        # INT (service & delivery counts)
        "intra_province": caster.to_int(raw_row.get("intra_province")),
        "international": caster.to_int(raw_row.get("international")),
        
        # INT (service types)
        "ser_c": caster.to_int(raw_row.get("ser_c")),
        "ser_e": caster.to_int(raw_row.get("ser_e")),
        "ser_m": caster.to_int(raw_row.get("ser_m")),
        "ser_p": caster.to_int(raw_row.get("ser_p")),
        "ser_r": caster.to_int(raw_row.get("ser_r")),
        "ser_u": caster.to_int(raw_row.get("ser_u")),
        "ser_l": caster.to_int(raw_row.get("ser_l")),
        "ser_q": caster.to_int(raw_row.get("ser_q")),
        
        # INT (issue counts)
        "delay_day": caster.to_int(raw_row.get("delay_day")),
        "delay_count": caster.to_int(raw_row.get("delay_count")),
        "nodone": caster.to_int(raw_row.get("nodone")),
        "refunded": caster.to_int(raw_row.get("refunded")),
        "noaccepted": caster.to_int(raw_row.get("noaccepted")),
        "lost_order": caster.to_int(raw_row.get("lost_order")),
        "lastday": caster.to_int(raw_row.get("lastday")),
        "noservice": caster.to_int(raw_row.get("noservice")),
        
        # DECIMAL scores
        "dev_item": caster.to_decimal(raw_row.get("dev_item"), 3),
        "order_score": caster.to_decimal(raw_row.get("order_score"), 3),
        "satisfaction_score": caster.to_decimal(raw_row.get("satisfaction_score"), 3),
        
        # INT (complaint counts)
        "total_complaint": caster.to_int(raw_row.get("total_complaint")),
        "complaint114": caster.to_int(raw_row.get("complaint114")),
        "complaint115": caster.to_int(raw_row.get("complaint115")),
        "complaint116": caster.to_int(raw_row.get("complaint116")),
        "complaint134": caster.to_int(raw_row.get("complaint134")),
        "complaint194": caster.to_int(raw_row.get("complaint194")),
        "complaint554": caster.to_int(raw_row.get("complaint554")),
        "complaint595": caster.to_int(raw_row.get("complaint595")),
        "complaint314": caster.to_int(raw_row.get("complaint314")),
        "complaint594": caster.to_int(raw_row.get("complaint594")),
        "complaint274": caster.to_int(raw_row.get("complaint274")),
        "complaint614": caster.to_int(raw_row.get("complaint614")),
        "complaint654": caster.to_int(raw_row.get("complaint654")),
        "complaint234": caster.to_int(raw_row.get("complaint234")),
        "complaint174": caster.to_int(raw_row.get("complaint174")),
        
        # TIMESTAMPTZ
        "updated_at": caster.to_timestamp(raw_row.get("updated_at")),
    }

def transform_cas_info_row(
    raw_row: Dict[str, Any], encrypto: Optional[CustomerEncryption] = None
) -> Optional[Dict[str, Any]]:
    """
    Transform 1 row từ cas_info CSV.
    Dùng SafeTypeCaster để cast TEXT → INT/TIMESTAMP an toàn.

    CSV header:
      cms_code_enc;crm_code_enc;cus_province;contract_service;tenure;
      custype;customer_update_date;contract_classify;contract_sig_first;
      contract_mgr_org;cus_poscode
    """
    if not raw_row:
        return None

    caster = SafeTypeCaster
    
    # DEBUG: Log raw datetime values for troubleshooting
    import logging
    logger = logging.getLogger(__name__)
    
    customer_update_date_raw = raw_row.get("customer_update_date")
    contract_sig_first_raw = raw_row.get("contract_sig_first")
    
    customer_update_date_ts = caster.to_timestamp(customer_update_date_raw)
    contract_sig_first_ts = caster.to_timestamp(contract_sig_first_raw)
    
    # Log only if transformation fails
    if customer_update_date_raw and not customer_update_date_ts:
        logger.warning(f"[CAS_INFO] Failed to parse customer_update_date: '{customer_update_date_raw}'")
    if contract_sig_first_raw and not contract_sig_first_ts:
        logger.warning(f"[CAS_INFO] Failed to parse contract_sig_first: '{contract_sig_first_raw}'")
    
    return {
        # VARCHAR(100)
        "cms_code_enc": caster.clean_string(raw_row.get("cms_code_enc")),
        "crm_code_enc": caster.clean_string(raw_row.get("crm_code_enc")),
        
        # BIGINT
        "cus_province": caster.to_int(raw_row.get("cus_province")),
        "contract_service": caster.to_int(raw_row.get("contract_service")),
        "tenure": caster.to_int(raw_row.get("tenure")),
        "custype": caster.to_int(raw_row.get("custype")),
        "contract_classify": caster.to_int(raw_row.get("contract_classify")),
        "contract_mgr_org": caster.to_int(raw_row.get("contract_mgr_org")),
        "cus_poscode": caster.to_int(raw_row.get("cus_poscode")),
        
        # TIMESTAMPTZ
        "customer_update_date": customer_update_date_ts,
        "contract_sig_first": contract_sig_first_ts,
    }

def transform_cms_complaint_row(
    raw_row: Dict[str, Any],
    encrypto: Optional[CustomerEncryption] = None,
) -> Optional[Dict[str, Any]]:
    """
    Transform 1 dòng cms_complaint.
    Dùng SafeTypeCaster để cast TEXT → INT/TIMESTAMP an toàn.

    CSV header:
      cms_code_enc;item_code;create_complaint_date;exp_complaint_date;close_complaint_date;
      delay_complaint;complaint_code;complaint_content;complaint_content_bit;
      complaint_update_date;etl_date

    Lưu ý:
      - CSV có thể dùng cms_code (không _enc), transform có fallback map thành cms_code_enc
      - Nếu cần mã hoá thật, có thể dùng encrypto.encode()
    """
    if not raw_row:
        return None

    caster = SafeTypeCaster
    
    return {
        # VARCHAR(20) - Normalized key = cms_code_enc (từ CSV cms_code)
        "cms_code_enc": caster.clean_string(raw_row.get("cms_code_enc")),
        "item_code": caster.clean_string(raw_row.get("item_code") or raw_row.get("item_code_enc")),
        
        # TIMESTAMPTZ
        "create_complaint_date": caster.to_timestamp(raw_row.get("create_complaint_date")),
        "exp_complaint_date": caster.to_timestamp(raw_row.get("exp_complaint_date")),
        "close_complaint_date": caster.to_timestamp(raw_row.get("close_complaint_date")),
        "complaint_update_date": caster.to_timestamp(raw_row.get("complaint_update_date")),
        "etl_date": caster.to_timestamp(raw_row.get("etl_date")),
        
        # INT
        "delay_complaint": caster.to_int(raw_row.get("delay_complaint")),
        "complaint_code": caster.to_int(raw_row.get("complaint_code")),
        "complaint_content_bit": caster.to_int(raw_row.get("complaint_content_bit")),
        
        # TEXT
        "complaint_content": caster.clean_string(raw_row.get("complaint_content")),
    }




