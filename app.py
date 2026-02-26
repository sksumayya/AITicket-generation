from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime
import random
import re
import joblib
from scipy.sparse import hstack
import spacy
from models.models import db, User, Ticket, TicketHistory,ContactMessage
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper function for AI categorization
def ai_categorize_ticket(description):
    """Simulate AI categorization based on keywords"""
    description_lower = description.lower()
    
    # Define categories and their keywords
    categories = {
        'technical': ['error', 'bug', 'crash', 'slow', 'login', 'password', 'install', 'update'],
        'billing': ['payment', 'invoice', 'charge', 'refund', 'bill', 'subscription', 'price'],
        'feature': ['request', 'suggestion', 'idea', 'improvement', 'new feature', 'enhancement'],
        'account': ['account', 'profile', 'settings', 'delete', 'suspended', 'reactivate'],
        'general': ['question', 'help', 'support', 'information', 'how to', 'tutorial']
    }
    
    # Calculate scores for each category
    scores = {}
    for category, keywords in categories.items():
        scores[category] = sum(1 for keyword in keywords if keyword in description_lower)
    
    # Determine primary category
    primary_category = max(scores, key=scores.get)
    confidence = scores[primary_category] / len(description.split()) * 10
    confidence = min(confidence, 1.0)  # Cap at 1.0
    
    return primary_category, confidence


# -------------------------------
# Load ML model and vectorizers (if available)
# -------------------------------
try:
    model = joblib.load('results/saved_model/category_model.pkl')
    word_tfidf = joblib.load('results/saved_model/word_vectorizer.pkl')
    char_tfidf = joblib.load('results/saved_model/char_vectorizer.pkl')
    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        nlp = None
    ML_AVAILABLE = True
except Exception:
    model = None
    word_tfidf = None
    char_tfidf = None
    nlp = None
    ML_AVAILABLE = False


# -------------------------------
# Priority logic (same as training)
# -------------------------------
def predict_priority(text):
    t = text.lower()
    if any(w in t for w in ["urgent", "crash", "down", "outage", "not working"]):
        return "High"
    if any(w in t for w in ["slow", "delay", "issue", "problem"]):
        return "Medium"
    return "Low"


# -------------------------------
# Entity Extraction
# -------------------------------
def extract_entities(text):
    if nlp is None:
        return []
    return [(ent.text, ent.label_) for ent in nlp(text).ents]


# -------------------------------
# SINGLE TEXT PREDICTION
# -------------------------------
def predict_ticket(text):
    # If ML assets available, use them; otherwise fallback to heuristic
    if ML_AVAILABLE and model is not None and word_tfidf is not None and char_tfidf is not None:
        w = word_tfidf.transform([text])
        c = char_tfidf.transform([text])
        x = hstack([w, c])
        try:
            category = model.predict(x)[0]
        except Exception:
            category = ai_categorize_ticket(text)[0]
    else:
        category = ai_categorize_ticket(text)[0]

    priority = predict_priority(text)
    entities = extract_entities(text)

    return {
        "title": text.split(".")[0][:60],
        "description": text,
        "category": category,
        "priority": priority,
        "type": "Incident" if priority == "High" else "Request",
        "entities": entities
    }

# Helper function for AI priority
def ai_assign_priority(description, category):
    """Simulate AI priority assignment"""
    urgent_keywords = ['urgent', 'emergency', 'critical', 'broken', 'not working', 'down', 'outage']
    high_keywords = ['important', 'need help', 'asap', 'immediately', 'blocked']
    
    description_lower = description.lower()
    
    if any(keyword in description_lower for keyword in urgent_keywords):
        return 'high', 0.9
    elif any(keyword in description_lower for keyword in high_keywords):
        return 'medium', 0.7
    elif category == 'technical':
        return 'medium', 0.6
    else:
        return 'low', 0.5

# Helper function for AI insights
def ai_generate_insights(description, category, priority):
    """Generate AI insights for the ticket"""
    insights = [
        f"Based on the description, this appears to be a {category} issue. ",
        f"The AI has assigned {priority} priority based on urgency indicators. ",
        "Recommended resolution time: ",
        "Similar past tickets were resolved within 24-48 hours. ",
        "This will be routed to the appropriate department automatically. "
    ]
    
    if 'error' in description.lower():
        insights.append("Error patterns detected - technical team has been notified. ")
    if 'payment' in description.lower():
        insights.append("Billing-related queries typically resolve within 1 business day. ")
    
    return "".join(insights)

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        # Get recent tickets for logged in users
        recent_tickets = Ticket.query.filter_by(user_id=current_user.id)\
            .order_by(Ticket.created_at.desc())\
            .limit(5)\
            .all()
        
        # Get statistics for logged in users
        total_tickets = Ticket.query.filter_by(user_id=current_user.id).count()
        open_tickets = Ticket.query.filter_by(user_id=current_user.id, status='open').count()
        in_progress_tickets = Ticket.query.filter_by(user_id=current_user.id, status='in_progress').count()
        resolved_tickets = Ticket.query.filter_by(user_id=current_user.id, status='resolved').count()
        
        return render_template('index.html',
                             recent_tickets=recent_tickets,
                             total_tickets=total_tickets,
                             open_tickets=open_tickets,
                             in_progress_tickets=in_progress_tickets,
                             resolved_tickets=resolved_tickets)
    
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get statistics
    total_tickets = Ticket.query.filter_by(user_id=current_user.id).count()
    open_tickets = Ticket.query.filter_by(user_id=current_user.id, status='open').count()
    in_progress_tickets = Ticket.query.filter_by(user_id=current_user.id, status='in_progress').count()
    resolved_tickets = Ticket.query.filter_by(user_id=current_user.id, status='resolved').count()
    
    # Get recent tickets
    recent_tickets = Ticket.query.filter_by(user_id=current_user.id)\
        .order_by(Ticket.created_at.desc())\
        .limit(5)\
        .all()
    
    return render_template('dashboard.html',
                         total_tickets=total_tickets,
                         open_tickets=open_tickets,
                         in_progress_tickets=in_progress_tickets,
                         resolved_tickets=resolved_tickets,
                         recent_tickets=recent_tickets)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists
        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/create-ticket', methods=['GET', 'POST'])
@login_required
def create_ticket():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category', '')
        priority = request.form.get('priority', 'medium')
        
        # AI/ML Analysis (use ML if available)
        ml_result = predict_ticket(description)
        ai_category = ml_result.get('category')
        ai_priority = ml_result.get('priority')
        ai_insights = ai_generate_insights(description, ai_category, ai_priority)
        ai_confidence = 0.95 if ML_AVAILABLE else 0.6
        ai_priority_score = 0.9 if ai_priority == 'High' else (0.7 if ai_priority == 'Medium' else 0.5)

        # Use AI/ML suggestions if user didn't specify
        if not category:
            category = ai_category
        if not priority:
            priority = ai_priority
        
        # Generate ticket ID
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        
        # Create ticket
        ticket = Ticket(
            ticket_id=ticket_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            user_id=current_user.id,
            ai_category=ai_category,
            ai_priority=ai_priority,
            ai_confidence=ai_confidence,
            ai_insights=ai_insights
        )
        
        db.session.add(ticket)
        
        # Add to history
        history = TicketHistory(
            ticket=ticket,
            action='Ticket Created',
            details=f'Ticket created by {current_user.username}',
            changed_by=current_user.username
        )
        db.session.add(history)
        
        db.session.commit()
        
        flash(f'Ticket {ticket_id} created successfully!', 'success')
        return redirect(url_for('ticket_history'))
    
    return render_template('create_ticket.html')

@app.route('/ticket-history')
@login_required
def ticket_history():
    # Get search and filter parameters
    search = request.args.get('search', '')
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    priority = request.args.get('priority', '')
    
    # Build query
    query = Ticket.query.filter_by(user_id=current_user.id)
    
    if search:
        query = query.filter(
            (Ticket.title.ilike(f'%{search}%')) | 
            (Ticket.description.ilike(f'%{search}%')) |
            (Ticket.ticket_id.ilike(f'%{search}%'))
        )
    
    if category:
        query = query.filter_by(category=category)
    
    if status:
        query = query.filter_by(status=status)
    
    if priority:
        query = query.filter_by(priority=priority)
    
    # Get tickets
    tickets = query.order_by(Ticket.created_at.desc()).all()
    
    # Get unique categories for filter dropdown
    categories = db.session.query(Ticket.category).filter_by(user_id=current_user.id).distinct().all()
    categories = [c[0] for c in categories]
    
    return render_template('ticket_history.html',
                         tickets=tickets,
                         categories=categories,
                         search=search,
                         selected_category=category,
                         selected_status=status,
                         selected_priority=priority)

@app.route('/ticket/<ticket_id>')
@login_required
def view_ticket(ticket_id):
    ticket = Ticket.query.filter_by(ticket_id=ticket_id, user_id=current_user.id).first_or_404()
    return render_template('view_ticket.html', ticket=ticket, get_estimated_resolution=get_estimated_resolution)

@app.route('/api/analyze-ticket', methods=['POST'])
@login_required
def analyze_ticket():
    """API endpoint for AI analysis"""
    data = request.get_json()
    description = data.get('description', '')
    
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    # Use ML prediction when available
    ml = predict_ticket(description)
    category = ml.get('category')
    priority = ml.get('priority')
    insights = ai_generate_insights(description, category, priority)
    confidence = 0.95 if ML_AVAILABLE else 0.6
    priority_score = 0.9 if priority == 'High' else (0.7 if priority == 'Medium' else 0.5)

    return jsonify({
        'ai_category': category,
        'ai_confidence': round(confidence, 2),
        'ai_priority': priority,
        'ai_priority_score': round(priority_score, 2),
        'ai_insights': insights,
        'entities': ml.get('entities', []),
        'type': ml.get('type'),
        'title': ml.get('title'),
        'estimated_resolution': get_estimated_resolution(category, priority)
    })

def get_estimated_resolution(category, priority):
    """Get estimated resolution time based on category and priority"""
    estimates = {
        ('technical', 'high'): '2-4 hours',
        ('technical', 'medium'): '24-48 hours',
        ('technical', 'low'): '3-5 days',
        ('billing', 'high'): '4-6 hours',
        ('billing', 'medium'): '24 hours',
        ('billing', 'low'): '2-3 days',
        ('feature', 'high'): '7-14 days',
        ('feature', 'medium'): 'Next sprint',
        ('feature', 'low'): 'Backlog',
        ('account', 'high'): '1-2 hours',
        ('account', 'medium'): '12-24 hours',
        ('account', 'low'): '2-3 days',
        ('general', 'high'): '4-8 hours',
        ('general', 'medium'): '24-48 hours',
        ('general', 'low'): '3-5 days'
    }
    
    return estimates.get((category, priority), '24-48 hours')

@app.route('/about')
def about():
    return render_template('about.html')

# API endpoints for dashboard
@app.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    total_tickets = Ticket.query.filter_by(user_id=current_user.id).count()
    open_tickets = Ticket.query.filter_by(user_id=current_user.id, status='open').count()
    in_progress_tickets = Ticket.query.filter_by(user_id=current_user.id, status='in_progress').count()
    resolved_tickets = Ticket.query.filter_by(user_id=current_user.id, status='resolved').count()
    
    return jsonify({
        'total_tickets': total_tickets,
        'open_tickets': open_tickets,
        'in_progress_tickets': in_progress_tickets,
        'resolved_tickets': resolved_tickets,
        'ai_accuracy': 96  # Mock AI accuracy
    })
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        email = request.form.get('email')
        phone = request.form.get('phone')
        subject = request.form.get('subject')
        message = request.form.get('message')

        new_message = ContactMessage(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            subject=subject,
            message=message
        )

        db.session.add(new_message)
        db.session.commit()

        return jsonify({'status': 'success'})

    return render_template('contact.html')
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')
@app.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        current_user.username = request.form.get('username')
        current_user.email = request.form.get('email')
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('edit_profile.html')


@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')

        if not current_user.check_password(old_password):
            flash("Old password is incorrect!", "danger")
            return redirect(url_for('change_password'))

        current_user.set_password(new_password)
        db.session.commit()

        flash("Password changed successfully!", "success")
        return redirect(url_for('profile'))

    return render_template('change_password.html')
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    return render_template('settings.html')
@app.route('/deactivate-account', methods=['POST'])
@login_required
def deactivate_account():
    logout_user()
    flash("Your account has been deactivated.", "info")
    return redirect(url_for('home'))


@app.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    user = User.query.get(current_user.id)  # Get real user object

    logout_user()  # Logout first

    db.session.delete(user)
    db.session.commit()

    flash("Your account has been permanently deleted.", "danger")
    return redirect(url_for('home'))
@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/api")
def api():
    return render_template("api.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/careers")
def careers():
    return render_template("careers.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")
@app.route('/blog/<int:blog_id>')
def blog_detail(blog_id):

    blogs = {
        1: {
            "title": "How AI Improves Ticket Categorization",
            "content": """
            Our AI-powered ticket system uses Natural Language Processing (NLP)
            to analyze ticket descriptions. It automatically detects keywords,
            patterns, and context to assign accurate categories.

            This reduces manual effort and improves workflow efficiency.
            """
        },
        2: {
            "title": "Advanced Analytics Dashboard",
            "content": """
            The analytics dashboard provides insights into ticket resolution rates,
            AI accuracy, performance trends, and workload distribution.
            """
        },
        3: {
            "title": "Reducing Ticket Resolution Time",
            "content": """
            By combining automation and AI categorization, support teams can
            significantly reduce response and resolution times.
            """
        },
        4: {
            "title": "Smart Priority Prediction with AI",
            "content": """
            Our BERT-based model predicts ticket priority levels such as Low,
            Medium, High, and Critical to ensure urgent issues are addressed first.
            """
        },
        5: {
            "title": "How Enterprises Use AI Ticket Systems",
            "content": """
            Enterprises use AI ticket systems to streamline support operations,
            reduce manual classification, and improve SLA compliance.
            """
        },
        6: {
            "title": "AI Ticket Platform Roadmap",
            "content": """
            Future improvements include enhanced model accuracy,
            multi-language support, and real-time AI recommendations.
            """
        }
    }

    blog = blogs.get(blog_id)

    if not blog:
        return "Blog not found", 404

    return render_template("blog_detail.html", blog=blog)
if __name__ == '__main__':
    app.run(debug=True)