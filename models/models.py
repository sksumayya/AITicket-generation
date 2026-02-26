from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    role = db.Column(db.String(20), default='user')  # user, admin, support
    is_active = db.Column(db.Boolean, default=True)
    # Relationships
    tickets = db.relationship('Ticket', backref='author', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Ticket(db.Model):
    __tablename__ = 'tickets'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.String(50), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    priority = db.Column(db.String(20), nullable=False)  # low, medium, high
    status = db.Column(db.String(20), default='open')  # open, in_progress, resolved, closed
    ai_category = db.Column(db.String(50))
    ai_priority = db.Column(db.String(20))
    ai_confidence = db.Column(db.Float)
    ai_insights = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    def __repr__(self):
        return f'<Ticket {self.ticket_id}>'

class TicketHistory(db.Model):
    __tablename__ = 'ticket_history'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('tickets.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    changed_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    ticket = db.relationship('Ticket', backref=db.backref('history', lazy=True))
    
    def __repr__(self):
        return f'<TicketHistory {self.ticket_id}: {self.action}>'
    
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
