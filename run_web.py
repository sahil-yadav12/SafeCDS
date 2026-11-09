import uvicorn
import os
import sys

def main():
    print("=" * 65)
    print("      SAFECDS NEURO-SYMBOLIC CLINICAL DECISION SUPPORT      ")
    print("=" * 65)
    print("Starting SafeCDS Web Application Server...")
    print("Access the dashboard locally at: http://localhost:8000")
    print("=" * 65)
    
    # Ensure port 8000 is used
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
