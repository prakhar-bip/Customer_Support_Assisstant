"""
Unit Tests for Preprocessing Module (src/preprocessing.py).
"""

import pytest
from src.preprocessing import clean_customer_message

def test_clean_customer_message_basic():
    text = "   Hello @AmazonHelp! Where is my package???   "
    cleaned = clean_customer_message(text)
    assert cleaned == "Hello @AmazonHelp! Where is my package???"

def test_clean_customer_message_whitespace_and_newlines():
    text = "Line 1\n\nLine 2\t\twith   tabs   and   spaces"
    cleaned = clean_customer_message(text)
    assert cleaned == "Line 1 Line 2 with tabs and spaces"

def test_clean_customer_message_null_and_empty():
    assert clean_customer_message(None) == ""
    assert clean_customer_message("") == ""
    assert clean_customer_message("   ") == ""

def test_clean_customer_message_control_chars():
    text = "Account compromised \x00\x08\x1b"
    cleaned = clean_customer_message(text)
    assert cleaned == "Account compromised"

def test_clean_customer_message_preserves_emojis_and_casing():
    text = "DELIVERY FAILED! 😡📦 Please help ASAP"
    cleaned = clean_customer_message(text)
    assert "😡📦" in cleaned
    assert "DELIVERY FAILED!" in cleaned
