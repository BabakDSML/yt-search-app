import requests
import json
import polars as pl
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import os

# ✅ NEW getVideoIDs() — uses PlaylistItems API
def getVideoIDs():
    """
    Function to return all video IDs from channel uploads playlist (better than Search API)

    Dependencies: 
        - get_video_records_from_playlist()
    """

    # Set your uploads playlist ID here
    playlist_id = "UUFtEEv80fQVKkD4h1PF-Xqw"  # this is your uploads playlist ID

    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    my_key = os.getenv('YT_API_KEY')

    params = {
        "key": my_key,
        "playlistId": playlist_id,
        "part": "snippet",
        "maxResults": 50
    }

    # Inner function to extract video records
    def get_video_records_from_playlist(response_json):
        return [
            {
                "video_id": item["snippet"]["resourceId"]["videoId"],
                "datetime": item["snippet"]["publishedAt"],
                "title": item["snippet"]["title"]
            }
            for item in response_json["items"]
        ]

    video_record_list = []
    page_token = None

    # Loop through pages
    while True:
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print("Error:", response.text)
            break

        response_json = response.json()
        video_record_list += get_video_records_from_playlist(response_json)

        page_token = response_json.get("nextPageToken")
        if not page_token:
            break

    # Save parquet
    pl.DataFrame(video_record_list).write_parquet('data/video-ids.parquet')

# --- The rest of your original functions stay the same ---

def extractTranscriptText(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary
    """
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)


def getVideoTranscripts():
    """
        Function to extract transcripts for all video IDs stored in "data/video-ids.parquet"
    """
    df = pl.read_parquet('data/video-ids.parquet')

    transcript_text_list = []

    for i in range(len(df)):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
            transcript_text = extractTranscriptText(transcript)
        except:
            transcript_text = "n/a"
        
        transcript_text_list.append(transcript_text)

    df = df.with_columns(pl.Series(name="transcript", values=transcript_text_list))
    df.write_parquet('data/video-transcripts.parquet')


def handleSpecialStrings(df: pl.dataframe.frame.DataFrame) -> pl.dataframe.frame.DataFrame:
    """
        Function to replace special character strings in video transcripts and titles
    """
    special_strings = ['&#39;', '&amp;', 'sha ']
    special_string_replacements = ["'", "&", "Shaw "]

    for i in range(len(special_strings)):
        df = df.with_columns(df['title'].str.replace(special_strings[i], special_string_replacements[i]).alias('title'))
        df = df.with_columns(df['transcript'].str.replace(special_strings[i], special_string_replacements[i]).alias('transcript'))

    return df


def setDatatypes(df: pl.dataframe.frame.DataFrame) -> pl.dataframe.frame.DataFrame:
    """
        Function to change data types of columns in polars data frame
    """
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
    return df


def transformData():
    """
        Function to preprocess video data
    """
    df = pl.read_parquet('data/video-transcripts.parquet')
    df = handleSpecialStrings(df)
    df = setDatatypes(df)
    df.write_parquet('data/video-transcripts.parquet')


def createTextEmbeddings():
    """
        Function to generate text embeddings of video titles and transcripts
    """
    df = pl.read_parquet('data/video-transcripts.parquet')

    model = SentenceTransformer('all-MiniLM-L6-v2')

    column_name_list = ['title', 'transcript']

    for column_name in column_name_list:
        embedding_arr = model.encode(df[column_name].to_list())

        schema_dict = {column_name+'_embedding-'+str(i): float for i in range(embedding_arr.shape[1])}
        df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

        df = pl.concat([df, df_embedding], how='horizontal')

    df.write_parquet('data/video-index.parquet')
