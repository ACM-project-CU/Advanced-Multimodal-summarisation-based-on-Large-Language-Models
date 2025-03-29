import os
from text_summariser import summarize_video

def main():
    # Define the path to the video file
    video_path = os.path.join(os.path.dirname(__file__), "Global Warming 101.mp4")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print("This may take several minutes depending on the video length and complexity...")
    
    # Run the video summarization function
    summary = summarize_video(video_path)
    
    # Save the summary to a text file
    output_file = os.path.join(os.path.dirname(__file__), "video_summary_result.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nSummary completed and saved to: {output_file}")
    print("\nSummary preview:")
    # Print the first 500 characters of the summary as a preview
    print(summary[:500] + "..." if len(summary) > 500 else summary)

if __name__ == "__main__":
    main()