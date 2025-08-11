"""
GUI-Based Phishing Detection Tool for Kathmandu College Students
Final Year Dissertation Project
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

class PhishingDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Detection Tool - Sakchyam Karmacharya")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.model = None
        self.scaler = None
        self.df = None
        self.feature_imp = None
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_welcome_tab()
        self.create_data_tab()
        self.create_model_tab()
        self.create_results_tab()
        self.create_recommendations_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_welcome_tab(self):
        """Create the welcome tab with project information"""
        welcome_tab = ttk.Frame(self.notebook)
        self.notebook.add(welcome_tab, text="Welcome")
        
        # Add logo or banner
        try:
            # You can replace this with your actual logo
            logo_img = Image.new('RGB', (800, 150), color='white')
            logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(welcome_tab, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack(pady=10)
        except:
            pass
        
        # Project information
        info_frame = ttk.LabelFrame(welcome_tab, text="Project Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = """
        AI-Based Phishing Detection for Kathmandu College Students
        
        This tool helps detect phishing websites with a focus on patterns 
        relevant to Nepali students. The system uses machine learning to 
        analyze website characteristics and identify potential threats.
        
        Features:
        - Load and analyze website data
        - Train a phishing detection model
        - View model performance metrics
        - Get cybersecurity education recommendations
        
        Instructions:
        1. Go to the 'Data' tab to load or generate data
        2. Train the model in the 'Model' tab
        3. View results and recommendations in their respective tabs
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(padx=10, pady=10, anchor=tk.W)
        
        # Quick start button
        quick_start_btn = ttk.Button(welcome_tab, text="Quick Start - Load Default Data", 
                                   command=self.load_default_data)
        quick_start_btn.pack(pady=10)
    
    def create_data_tab(self):
        """Create the data tab for loading and viewing data"""
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Data")
        
        # Data source frame
        source_frame = ttk.LabelFrame(data_tab, text="Data Source")
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Data source options
        self.data_source = tk.StringVar(value="default")
        
        ttk.Radiobutton(source_frame, text="Load Default Dataset (UCI)", 
                       variable=self.data_source, value="default").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Load from CSV File", 
                       variable=self.data_source, value="csv").pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Generate Synthetic Data", 
                       variable=self.data_source, value="synthetic").pack(anchor=tk.W)
        
        # File selection for CSV
        self.csv_path = tk.StringVar()
        csv_frame = ttk.Frame(source_frame)
        csv_frame.pack(fill=tk.X, pady=5)
        ttk.Label(csv_frame, text="CSV File:").pack(side=tk.LEFT)
        ttk.Entry(csv_frame, textvariable=self.csv_path, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(csv_frame, text="Browse", command=self.browse_csv).pack(side=tk.LEFT)
        
        # Load data button
        ttk.Button(source_frame, text="Load Data", command=self.load_data).pack(pady=5)
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(data_tab, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data display
        self.data_tree = ttk.Treeview(preview_frame)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=y_scroll.set)
        
        x_scroll = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.configure(xscrollcommand=x_scroll.set)
        
        # Data info
        self.data_info = ttk.Label(data_tab, text="No data loaded")
        self.data_info.pack(pady=5)
    
    def create_model_tab(self):
        """Create the model tab for training and evaluation"""
        model_tab = ttk.Frame(self.notebook)
        self.notebook.add(model_tab, text="Model")
        
        # Model controls frame
        controls_frame = ttk.LabelFrame(model_tab, text="Model Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Train/test split slider
        ttk.Label(controls_frame, text="Test Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.test_size = tk.DoubleVar(value=0.3)
        ttk.Scale(controls_frame, from_=0.1, to=0.5, variable=self.test_size, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(controls_frame, textvariable=self.test_size).grid(row=0, column=2, padx=5, pady=5)
        
        # Model parameters
        ttk.Label(controls_frame, text="Estimators:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.n_estimators = tk.IntVar(value=150)
        ttk.Entry(controls_frame, textvariable=self.n_estimators, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(controls_frame, text="Max Depth:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_depth = tk.IntVar(value=12)
        ttk.Entry(controls_frame, textvariable=self.max_depth, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Preprocess Data", command=self.preprocess_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Evaluate Model", command=self.evaluate_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(model_tab, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Confusion matrix display
        self.confusion_canvas = None
        self.confusion_frame = ttk.Frame(results_frame)
        self.confusion_frame.pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """Create the results tab for viewing feature importance"""
        results_tab = ttk.Frame(self.notebook)
        self.notebook.add(results_tab, text="Feature Analysis")
        
        # Feature importance display
        feature_frame = ttk.LabelFrame(results_tab, text="Feature Importance")
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.feature_canvas = None
        self.feature_frame = ttk.Frame(feature_frame)
        self.feature_frame.pack(fill=tk.BOTH, expand=True)
        
        # Feature importance table
        self.feature_table = ttk.Treeview(feature_frame)
        self.feature_table.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        self.feature_table['columns'] = ('rank', 'feature', 'importance')
        self.feature_table.column('#0', width=0, stretch=tk.NO)
        self.feature_table.column('rank', anchor=tk.CENTER, width=50)
        self.feature_table.column('feature', anchor=tk.W, width=200)
        self.feature_table.column('importance', anchor=tk.CENTER, width=100)
        
        # Create headings
        self.feature_table.heading('rank', text='Rank', anchor=tk.CENTER)
        self.feature_table.heading('feature', text='Feature', anchor=tk.CENTER)
        self.feature_table.heading('importance', text='Importance', anchor=tk.CENTER)
        
        # Export button
        ttk.Button(feature_frame, text="Export Feature Importance", 
                  command=self.export_feature_importance).pack(pady=5)
    
    def create_recommendations_tab(self):
        """Create the recommendations tab"""
        rec_tab = ttk.Frame(self.notebook)
        self.notebook.add(rec_tab, text="Recommendations")
        
        # Recommendations display
        rec_frame = ttk.LabelFrame(rec_tab, text="Cybersecurity Education Recommendations")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.rec_text = scrolledtext.ScrolledText(rec_frame, wrap=tk.WORD)
        self.rec_text.pack(fill=tk.BOTH, expand=True)
        
        # Generate button
        ttk.Button(rec_frame, text="Generate Recommendations", 
                  command=self.generate_recommendations).pack(pady=5)
        
        # Export button
        ttk.Button(rec_frame, text="Export Recommendations", 
                  command=self.export_recommendations).pack(pady=5)
    
    def browse_csv(self):
        """Browse for CSV file"""
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.csv_path.set(filepath)
    
    def load_default_data(self):
        """Load the default dataset"""
        self.data_source.set("default")
        self.load_data()
    
    def load_data(self):
        """Load data based on selected source"""
        source = self.data_source.get()
        
        try:
            if source == "default":
                self.status_var.set("Loading default dataset...")
                self.root.update()
                
                # Load dataset from UCI repository (replace with local data later)
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
                self.df = pd.read_csv(url, skiprows=45, header=None)
                
                # Define column names based on dataset documentation
                columns = [
                    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
                    'Statistical_report', 'Result'
                ]
                self.df.columns = columns
                
                # Convert target to binary (phishing = 0, legitimate = 1)
                self.df['Result'] = self.df['Result'].map({-1: 0, 1: 1})
                
                # Add simulated Nepal-specific features (replace with actual data)
                np.random.seed(42)
                self.df['nepali_keyword_usage'] = np.random.uniform(0, 1, size=len(self.df))
                self.df['academic_theme'] = np.random.randint(0, 3, size=len(self.df))
                
                messagebox.showinfo("Success", "Default dataset loaded successfully!")
                
            elif source == "csv":
                filepath = self.csv_path.get()
                if not filepath:
                    messagebox.showerror("Error", "Please select a CSV file")
                    return
                
                self.status_var.set(f"Loading data from {filepath}...")
                self.root.update()
                
                self.df = pd.read_csv(filepath)
                messagebox.showinfo("Success", "CSV file loaded successfully!")
            
            elif source == "synthetic":
                self.status_var.set("Generating synthetic data...")
                self.root.update()
                
                # Generate synthetic data (simplified example)
                num_samples = 1000
                self.df = pd.DataFrame({
                    'URL_Length': np.random.randint(10, 200, size=num_samples),
                    'having_At_Symbol': np.random.randint(0, 2, size=num_samples),
                    'SSLfinal_State': np.random.randint(0, 2, size=num_samples),
                    'having_Sub_Domain': np.random.randint(0, 5, size=num_samples),
                    'nepali_keyword_usage': np.random.uniform(0, 1, size=num_samples),
                    'academic_theme': np.random.randint(0, 3, size=num_samples),
                    'Result': np.random.randint(0, 2, size=num_samples)
                })
                
                messagebox.showinfo("Success", f"Synthetic data with {num_samples} samples generated!")
            
            # Update data preview
            self.update_data_preview()
            self.status_var.set("Data loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def update_data_preview(self):
        """Update the data preview treeview"""
        if self.df is None:
            return
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set up columns
        self.data_tree['columns'] = list(self.df.columns)
        for col in self.df.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, minwidth=50)
        
        # Add data (limit to 100 rows for performance)
        sample_df = self.df.head(100)
        for _, row in sample_df.iterrows():
            self.data_tree.insert('', tk.END, values=list(row))
        
        # Update info label
        self.data_info.config(text=f"Loaded {len(self.df)} samples with {len(self.df.columns)} features")
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        if self.df is None:
            messagebox.showerror("Error", "No data loaded. Please load data first.")
            return
        
        try:
            self.status_var.set("Preprocessing data...")
            self.root.update()
            
            # Separate features and target
            X = self.df.drop('Result', axis=1)
            y = self.df['Result']
            
            # Normalize numerical features
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data into train and test sets
            test_size = self.test_size.get()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            messagebox.showinfo("Success", 
                              f"Data preprocessed successfully!\n"
                              f"Training samples: {len(self.X_train)}\n"
                              f"Test samples: {len(self.X_test)}")
            self.status_var.set("Data preprocessed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess data: {str(e)}")
            self.status_var.set("Error preprocessing data")
    
    def train_model(self):
        """Train the phishing detection model"""
        if not hasattr(self, 'X_train') or self.X_train is None:
            messagebox.showerror("Error", "Data not preprocessed. Please preprocess data first.")
            return
        
        try:
            self.status_var.set("Training model...")
            self.root.update()
            
            # Initialize model with parameters from GUI
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators.get(),
                max_depth=self.max_depth.get(),
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            messagebox.showinfo("Success", "Model trained successfully!")
            self.status_var.set("Model trained")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Error training model")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            messagebox.showerror("Error", "Model not trained. Please train the model first.")
            return
        
        try:
            self.status_var.set("Evaluating model...")
            self.root.update()
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Generate classification report
            report = classification_report(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Model Evaluation Report:\n")
            self.results_text.insert(tk.END, "========================\n\n")
            self.results_text.insert(tk.END, report)
            self.results_text.insert(tk.END, f"\nOverall Accuracy: {accuracy:.2%}\n")
            
            # Generate and display confusion matrix
            self.show_confusion_matrix(self.y_test, y_pred)
            
            # Analyze feature importance
            self.analyze_features()
            
            messagebox.showinfo("Success", "Model evaluated successfully!")
            self.status_var.set("Model evaluated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model: {str(e)}")
            self.status_var.set("Error evaluating model")
    
    def show_confusion_matrix(self, y_true, y_pred):
        """Display the confusion matrix in the GUI"""
        # Clear previous canvas if exists
        if self.confusion_canvas:
            self.confusion_canvas.get_tk_widget().destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Phishing', 'Legitimate'],
                    yticklabels=['Phishing', 'Legitimate'], ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        
        # Embed in Tkinter
        self.confusion_canvas = FigureCanvasTkAgg(fig, master=self.confusion_frame)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_features(self):
        """Analyze and display feature importance"""
        if self.model is None:
            return
        
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = self.df.drop('Result', axis=1).columns
            
            # Create importance dataframe
            self.feature_imp = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            
            # Update feature importance table
            self.update_feature_table()
            
            # Display feature importance plot
            self.show_feature_importance()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze features: {str(e)}")
    
    def update_feature_table(self):
        """Update the feature importance table"""
        if self.feature_imp is None:
            return
        
        # Clear existing data
        for item in self.feature_table.get_children():
            self.feature_table.delete(item)
        
        # Add new data
        for i, (_, row) in enumerate(self.feature_imp.iterrows(), 1):
            self.feature_table.insert('', tk.END, values=(i, row['Feature'], f"{row['Importance']:.4f}"))
    
    def show_feature_importance(self):
        """Display the feature importance plot"""
        if self.feature_imp is None:
            return
        
        # Clear previous canvas if exists
        if self.feature_canvas:
            self.feature_canvas.get_tk_widget().destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', 
                   data=self.feature_imp.head(15), ax=ax)
        ax.set_title('Top 15 Important Features')
        
        # Embed in Tkinter
        self.feature_canvas = FigureCanvasTkAgg(fig, master=self.feature_frame)
        self.feature_canvas.draw()
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_recommendations(self):
        """Generate cybersecurity education recommendations"""
        if self.feature_imp is None:
            messagebox.showerror("Error", "No feature importance data available. Please evaluate the model first.")
            return
        
        try:
            # Get top features
            top_features = self.feature_imp.head(5)['Feature'].tolist()
            
            # Generate recommendations based on features
            recommendations = []
            
            if 'SSLfinal_State' in top_features:
                recommendations.append(
                    "1. Teach students to verify SSL certificates and look for HTTPS in URLs"
                )
            
            if 'URL_Length' in top_features:
                recommendations.append(
                    "2. Train students to identify suspiciously long or short URLs"
                )
            
            if 'having_Sub_Domain' in top_features:
                recommendations.append(
                    "3. Educate about subdomain spoofing (e.g., 'tribhuvan.university.fake.com')"
                )
            
            # Add Nepal-specific recommendations
            recommendations.extend([
                "4. Develop Nepali-language examples of phishing attempts",
                "5. Create simulations using local scam patterns (scholarship, job offers)",
                "6. Focus on mobile phishing (common on platforms like Khalti, eSewa)",
                "7. Conduct regular phishing simulation exercises",
                "8. Establish a cybersecurity awareness campaign in Social Engineering and Nepali"
            ])
            
            # Display recommendations
            self.rec_text.delete(1.0, tk.END)
            self.rec_text.insert(tk.END, "Cybersecurity Education Recommendations for Kathmandu Colleges:\n")
            self.rec_text.insert(tk.END, "===============================================================\n\n")
            for rec in recommendations:
                self.rec_text.insert(tk.END, rec + "\n\n")
            
            messagebox.showinfo("Success", "Recommendations generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {str(e)}")
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Model or scaler not available. Please train the model first.")
            return
        
        try:
            # Ask for directory
            dir_path = filedialog.askdirectory(title="Select directory to save model")
            if not dir_path:
                return
            
            # Create model_artifacts directory if it doesn't exist
            save_path = os.path.join(dir_path, 'model_artifacts')
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, os.path.join(save_path, 'phishing_model.pkl'))
            
            # Save scaler
            joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
            
            messagebox.showinfo("Success", f"Model and artifacts saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def export_feature_importance(self):
        """Export feature importance to CSV"""
        if self.feature_imp is None:
            messagebox.showerror("Error", "No feature importance data available.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save feature importance as"
            )
            
            if file_path:
                self.feature_imp.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Feature importance saved to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export feature importance: {str(e)}")
    
    def export_recommendations(self):
        """Export recommendations to text file"""
        if not self.rec_text.get(1.0, tk.END).strip():
            messagebox.showerror("Error", "No recommendations available.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save recommendations as"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.rec_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Recommendations saved to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export recommendations: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = PhishingDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()