import phonenumbers
from phonenumbers import NumberParseException
from typing import Optional


def normalize_phone_number(number: str, default_region: str = "DE") -> Optional[str]:
    """
    Normalize phone number to E.164 format (+49...).

    Args:
        number: Raw phone number string
        default_region: Default country code (DE, AT, CH)

    Returns:
        Normalized E.164 format or None if invalid
    """
    if not number:
        return None

    try:
        parsed = phonenumbers.parse(number, default_region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed,
                phonenumbers.PhoneNumberFormat.E164
            )
    except NumberParseException:
        pass

    return None


def format_phone_number(number: str, format_type: str = "international") -> Optional[str]:
    """
    Format phone number for display.

    Args:
        number: Phone number (any format)
        format_type: "international", "national", or "e164"

    Returns:
        Formatted phone number or None if invalid
    """
    if not number:
        return None

    try:
        parsed = phonenumbers.parse(number, "DE")

        if not phonenumbers.is_valid_number(parsed):
            return number  # Return original if can't parse

        format_map = {
            "international": phonenumbers.PhoneNumberFormat.INTERNATIONAL,
            "national": phonenumbers.PhoneNumberFormat.NATIONAL,
            "e164": phonenumbers.PhoneNumberFormat.E164
        }

        fmt = format_map.get(format_type, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return phonenumbers.format_number(parsed, fmt)

    except NumberParseException:
        return number


def validate_phone_number(number: str, region: str = "DE") -> bool:
    """
    Validate if phone number is valid for region.

    Args:
        number: Phone number string
        region: Country code to validate against

    Returns:
        True if valid, False otherwise
    """
    if not number:
        return False

    try:
        parsed = phonenumbers.parse(number, region)
        return phonenumbers.is_valid_number(parsed)
    except NumberParseException:
        return False


def is_mobile_number(number: str, region: str = "DE") -> bool:
    """
    Check if phone number is a mobile number.

    Args:
        number: Phone number string
        region: Default region

    Returns:
        True if mobile, False otherwise
    """
    if not number:
        return False

    try:
        parsed = phonenumbers.parse(number, region)
        number_type = phonenumbers.number_type(parsed)
        return number_type == phonenumbers.PhoneNumberType.MOBILE
    except NumberParseException:
        return False
