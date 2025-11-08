# app.py

from enhanced_severity_mapper import EnhancedSeverityMapper

def main():
    print("Starting enhanced severity mapping processing...")
    severity_mapper = EnhancedSeverityMapper()
    severity_mapper.process_unprocessed_incidents()
    print("Processing complete.")

if __name__ == "__main__":
    main()
