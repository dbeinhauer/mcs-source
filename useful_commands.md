# Start SSH with port open
```bash
ssh -L 8080:localhost:8080 neuro
```

# Start Jupyter Lab on port
```bash
jupyter lab --no-browser --port=8080
```

# Open directory in browser
```bash
python -m http.server 8081
```