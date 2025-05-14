from email.message import EmailMessage
import smtplib
from typing import List
from app.config import settings

def send_email(subject: str, body: str, to_emails: List[str]):
    message = EmailMessage()
    message.set_content(body)
    message["Subject"] = subject
    message["From"] = settings.EMAIL_USER
    message["To"] = ", ".join(to_emails)

    with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
        server.login(settings.EMAIL_USER, settings.EMAIL_PASSWORD)
        server.send_message(message)
