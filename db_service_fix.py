import os
import json
import hashlib
import datetime
import psycopg2
import numpy as np
import openai
from dotenv import load_dotenv
import yaml

class DatabaseService:
    def __init__(self):
        """Initialize the database service with connection parameters from .env"""
        load_dotenv()
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_schema = os.getenv("DB_SCHEMA")
        self.db_table = os.getenv("DB_TABLE")
        self.connection = None
        self.cursor = None
        self.yaml_config = self._load_yaml_config()
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("WARNING: OPENAI_API_KEY not found. Embeddings will not work.")
    
    def _load_yaml_config(self):
        """Load YAML configuration for risk scoring"""
        try:
            # First try to load from a real_estate.yaml file in config directory
            if os.path.exists('config/real_estate.yaml'):
                with open('config/real_estate.yaml', 'r') as file:
                    return yaml.safe_load(file)
            
            # If file doesn't exist, return default configuration
            return {
                'risk_thresholds': {
                    'low': 30,
                    'moderate': 70
                },
                'risk_factors': {
                    'flood_risk_score': {
                        'weight': 30,
                        'threshold': 20,
                        'impact': 'high'
                    },
                    'outdated_tax_delta': {
                        'weight': 20,
                        'threshold': 5.0,
                        'impact': 'medium'
                    },
                    'infrastructure_opacity': {
                        'weight': 15,
                        'threshold': 'Medium',
                        'impact': 'medium'
                    },
                    'regional_data_variation': {
                        'weight': 15,
                        'threshold': 'Medium',
                        'impact': 'medium'
                    },
                    'climate_score': {
                        'weight': 20,
                        'threshold': 300,
                        'impact': 'high'
                    }
                }
            }
        except Exception as e:
            print(f"Error loading YAML configuration: {str(e)}")
            # Return a minimal default configuration
            return {
                'risk_thresholds': {'low': 30, 'moderate': 70},
                'risk_factors': {}
            }
    
    def connect(self):
        """Establish a connection to the PostgreSQL database"""
        try:
            # Only create a new connection if we don't have one or it's closed
            if not self.connection or self.connection.closed:
                self.connection = psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
                self.cursor = self.connection.cursor()
                print("Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Database connection closed")
    
    def _get_embedding(self, text):
        """Generate an embedding for the given text using OpenAI's API"""
        try:
            # Handle newer OpenAI client format
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except (ImportError, AttributeError):
                # Fall back to older OpenAI client format
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-3-small",
                    api_key=self.openai_api_key
                )
                return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a zero vector of the right dimensionality (1536 for text-embedding-3-small)
            return [0.0] * 1536
    
    def _calculate_risk_score(self, analyzed_data):
        """Calculate risk score based on YAML configuration and analysis results"""
        if not self.yaml_config or not self.yaml_config.get('risk_factors'):
            # Default score if no configuration
            return 50
        
        # Start with a base score
        total_score = 0
        total_weight = 0
        
        # Map the analyzed_data keys to the config keys
        key_mapping = {
            'Flood Risk Score': 'flood_risk_score',
            'Outdated Tax Delta': 'outdated_tax_delta',
            'Infrastructure Opacity': 'infrastructure_opacity',
            'Regional Data Variation': 'regional_data_variation',
            'Climate Score': 'climate_score',
        }
        
        # Process each risk factor
        for analyzed_key, config_key in key_mapping.items():
            if config_key not in self.yaml_config['risk_factors']:
                continue
                
            factor_config = self.yaml_config['risk_factors'][config_key]
            factor_weight = factor_config.get('weight', 10)  # Default weight
            total_weight += factor_weight
            
            # Get the value from analyzed data
            raw_value = analyzed_data.get(analyzed_key, 'not found')
            if raw_value == 'not found':
                # Skip this factor if data is missing
                continue
                
            # Process different types of values
            if config_key == 'flood_risk_score':
                # Example: "28 (Moderate Risk)" -> Extract 28
                try:
                    numeric_part = raw_value.split('(')[0].strip()
                    score = int(numeric_part)
                    # Higher flood risk score = higher risk
                    factor_score = (score / 100) * 100  # Normalize to 0-100
                except (ValueError, AttributeError, IndexError):
                    factor_score = 50  # Default if parsing fails
            
            elif config_key == 'outdated_tax_delta':
                # Example: "4.1%" -> Extract 4.1
                try:
                    numeric_part = raw_value.strip('%')
                    delta = float(numeric_part)
                    # Higher tax delta = higher risk
                    threshold = factor_config.get('threshold', 5.0)
                    factor_score = min((delta / threshold) * 100, 100)
                except (ValueError, AttributeError):
                    factor_score = 50
            
            elif config_key in ['infrastructure_opacity', 'regional_data_variation']:
                # Example: "Low", "Medium", "High"
                mapping = {'Low': 25, 'Moderate': 50, 'Medium': 50, 'High': 75, 'Very High': 100}
                factor_score = mapping.get(raw_value, 50)
            
            elif config_key == 'climate_score':
                # Example: "72" -> Higher is better (less risk)
                try:
                    score = int(raw_value)
                    # Climate score is inverse - higher score = lower risk
                    factor_score = 100 - min((score / 100) * 100, 100)
                except (ValueError, AttributeError):
                    factor_score = 50
            
            else:
                # Default handling for unknown factors
                factor_score = 50
            
            # Add weighted score
            total_score += factor_score * factor_weight
        
        # Calculate final score, default to 50 if no factors were processed
        final_score = int(total_score / total_weight) if total_weight > 0 else 50
        
        return final_score
    
    def _calculate_checksum(self, file_path):
        """Calculate SHA-256 checksum of a file"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
                return file_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating checksum: {str(e)}")
            return None
    
    def create_table_if_not_exists(self):
        """Create the real estate documents table if it doesn't exist"""
        try:
            # Make sure we have a connection
            if not self.connect():
                return False
            
            # Check if table exists
            self.cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = %s AND tablename = %s
                );
            """, (self.db_schema, self.db_table))
            table_exists = self.cursor.fetchone()[0]
            
            if not table_exists:
                # Create pgvector extension if it doesn't exist
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create the table
                self.cursor.execute(f"""
                CREATE TABLE {self.db_schema}.{self.db_table} (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT UNIQUE NOT NULL,
                    extracted_text TEXT,
                    risk_score INTEGER,
                    embedding vector(1536),
                    mls_listing TEXT,
                    build_year INTEGER,
                    land_use_code TEXT,
                    flood_risk_score TEXT,
                    zoning_record TEXT,
                    outdated_tax_delta TEXT,
                    infrastructure_opacity TEXT,
                    regional_data_variation TEXT,
                    climate_score TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                self.connection.commit()
                print(f"Table {self.db_schema}.{self.db_table} created successfully")
            else:
                print(f"Table {self.db_schema}.{self.db_table} already exists")
            
            return True
        except Exception as e:
            print(f"Error creating table: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def store_document(self, file_path, analyzed_data, extracted_text, force_store=False):
        """Store document data in the database"""
        try:
            # Make sure we have a connection
            if not self.connect():
                return False
                
            # Ensure table exists
            if not self.create_table_if_not_exists():
                return False
                
            # Check if any key fields have 'not found' values, but only if not forcing storage
            if not force_store:
                important_fields = [
                    'Mls Listing', 'Build Year', 'Land Use Code', 'Flood Risk Score',
                    'Zoning Record', 'Outdated Tax Delta', 'Infrastructure Opacity',
                    'Regional Data Variation', 'Climate Score'
                ]
                
                missing_fields = [field for field in important_fields if analyzed_data.get(field, 'not found') == 'not found']
                
                if missing_fields:
                    print(f"Skipping database storage - found {len(missing_fields)} missing fields: {', '.join(missing_fields)}")
                    print("Will store data once all fields are populated by the user.")
                    return True
                
            file_name = os.path.basename(file_path)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(analyzed_data)
            
            # Generate embedding for the extracted text
            embedding = self._get_embedding(extracted_text)
            embedding_str = json.dumps(embedding)
            
            # Parse build year to integer if possible
            build_year = analyzed_data.get('Build Year', 'not found')
            if build_year != 'not found':
                try:
                    build_year = int(build_year)
                except ValueError:
                    pass
            
            # Check if document already exists
            self.cursor.execute(
                f"SELECT COUNT(*) FROM {self.db_schema}.{self.db_table} WHERE file_name = %s",
                (file_name,)
            )
            exists = self.cursor.fetchone()[0] > 0
            
            if exists:
                # Update existing document
                update_query = f"""
                UPDATE {self.db_schema}.{self.db_table}
                SET 
                    extracted_text = %s,
                    risk_score = %s,
                    embedding = %s::vector,
                    mls_listing = %s,
                    build_year = %s,
                    land_use_code = %s,
                    flood_risk_score = %s,
                    zoning_record = %s,
                    outdated_tax_delta = %s,
                    infrastructure_opacity = %s,
                    regional_data_variation = %s,
                    climate_score = %s,
                    created_at = NOW()
                WHERE file_name = %s
                """
                
                # Parse build year to integer if possible
                build_year = analyzed_data.get('Build Year', 'not found')
                try:
                    build_year = int(build_year) if build_year != 'not found' else None
                except ValueError:
                    build_year = None
                
                self.cursor.execute(
                    update_query,
                    (
                        extracted_text,
                        risk_score,
                        embedding_str,
                        analyzed_data.get('Mls Listing', 'not found'),
                        build_year,
                        analyzed_data.get('Land Use Code', 'not found'),
                        analyzed_data.get('Flood Risk Score', 'not found'),
                        analyzed_data.get('Zoning Record', 'not found'),
                        analyzed_data.get('Outdated Tax Delta', 'not found'),
                        analyzed_data.get('Infrastructure Opacity', 'not found'),
                        analyzed_data.get('Regional Data Variation', 'not found'),
                        analyzed_data.get('Climate Score', 'not found'),
                        file_name
                    )
                )
                print(f"Document {file_name} updated in database")
            else:
                # Insert new document
                insert_query = f"""
                INSERT INTO {self.db_schema}.{self.db_table}
                (file_name, extracted_text, risk_score, embedding,
                 mls_listing, build_year, land_use_code, flood_risk_score, zoning_record,
                 outdated_tax_delta, infrastructure_opacity, regional_data_variation,
                 climate_score)
                VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Parse build year to integer if possible
                build_year = analyzed_data.get('Build Year', 'not found')
                try:
                    build_year = int(build_year) if build_year != 'not found' else None
                except ValueError:
                    build_year = None
                
                self.cursor.execute(
                    insert_query,
                    (
                        file_name,
                        extracted_text,
                        risk_score,
                        embedding_str,
                        analyzed_data.get('Mls Listing', 'not found'),
                        build_year,
                        analyzed_data.get('Land Use Code', 'not found'),
                        analyzed_data.get('Flood Risk Score', 'not found'),
                        analyzed_data.get('Zoning Record', 'not found'),
                        analyzed_data.get('Outdated Tax Delta', 'not found'),
                        analyzed_data.get('Infrastructure Opacity', 'not found'),
                        analyzed_data.get('Regional Data Variation', 'not found'),
                        analyzed_data.get('Climate Score', 'not found')
                    )
                )
                print(f"Document {file_name} inserted into database")
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error storing document in database: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
        # IMPORTANT: We're not closing the connection in a finally block anymore!
