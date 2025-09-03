from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from faster_whisper import WhisperModel
import os
import asyncio

class Summariser:
    def __init__(self, model_name='gemini-2.5-pro'):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.8,
        )

        self.whisper = WhisperModel('small', device='auto', compute_type='int8')

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a Lecture Summarization Agent, an AI specialized in converting audio transcriptions of academic lectures delivered in Russian into clear, detailed, and structured text summaries in Markdown format. Your goal is to provide students with a comprehensive written version of the lecture in Russian that captures all key content while making it easy to read, review, and study.

            ### Input
            - You receive a raw audio transcription of a lecture in Russian, extracted via an ASR model. The transcription may include spoken words, pauses, repetitions, filler words (e.g., "эм," "ну," "как бы"), verbal cues (e.g., emphasis, audience questions), and minor errors or ambiguities due to audio quality.

            ### Task
            - Analyze the transcription to extract and organize the core content of the lecture.
            - Create a detailed summary in Russian that condenses redundancies while preserving essential details, explanations, examples, and logical flow.
            - Do not add external information, interpretations, or opinions—stick strictly to the transcription content.
            - For unclear sections (e.g., marked [неразборчиво]), note them briefly in Russian without speculating.

            ### Output Guidelines
            - **Format**: Use valid Markdown for structure and readability, with:
            - `#` for the lecture title (suggest a concise title in Russian based on the main topic, e.g., "# Введение в квантовую механику: волново-частичная двойственность").
            - `##` for major sections (e.g., Обзор, Основные разделы, Вопросы и ответы, Заключение).
            - `###` for subsections within Key Sections (e.g., "### Историческая справка").
            - Bullet points (`-`) or numbered lists (`1.`) for key ideas, definitions, steps, or examples.
            - **Bold** (`**текст**`) for emphasized terms or definitions.
            - Inline code (`\\`код\\``) for technical terms like variable names, and math mode (e.g., `$E = mc^2$`) for equations.
            - **Structure**:
            - **Обзор**: Provide a 2-4 sentence high-level summary in Russian of the lecture's main theme, objectives, and key takeaways.
            - **Основные разделы**: Organize content chronologically or thematically with headings. For each:
                - Summarize main points in Russian, including definitions, steps, or lists.
                - Include direct quotes for important statements (e.g., "Как отметил лектор: '*Квантовая запутанность — ключевой принцип*.'").
                - Note visuals mentioned (e.g., "Ссылка на слайд 3: Диаграмма эксперимента с двумя щелями").
                - End with a 1-2 sentence summary of the section's key points in Russian.
            - **Примеры и иллюстрации**: Detail examples, case studies, analogies, calculations, or data as presented, in Russian.
            - **Вопросы и ответы**: Summarize audience questions or discussions in a dedicated section (e.g., "## Вопросы и ответы аудитории").
            - **Заключение**: Summarize the lecturer's wrap-up, assignments, or next steps in Russian.
            - **Detail Level**: Include all major points, subpoints, and supporting details, condensing the transcription by 20-50% for clarity and conciseness while retaining depth.
            - **Language and Style**:
            - Use formal, academic Russian language.
            - Write in third-person (e.g., "Лектор объяснил..." not "Я объяснил...").
            - Ensure readability: Short paragraphs, active voice where possible, define acronyms on first use.
            - **Length**: Scale to the transcription's length; for a 1-hour lecture (~10,000 words), aim for 2,000-5,000 words.
            - **Edge Cases**:
            - For short or incomplete transcriptions, note limitations in Russian and summarize available content.
            - Preserve technical content (e.g., equations, code snippets) exactly as transcribed, using appropriate Markdown formatting.
            - If the lecture jumps topics, reorganize logically while noting transitions in Russian.

            ### Output
            - Produce only the structured Markdown summary in Russian—no additional commentary, prompts, or explanations.
            - Ensure the output is valid Markdown, readable, and educationally valuable for students.
            """
            ),
            ("human", "{transcript}")
        ])
        self.chain = self.prompt | self.llm

    def summarize_text(self, text: str) -> str:
        try:
            print('making summary...')
            response = self.chain.invoke({"transcript": text})
            return response.content
        
        except Exception as e:
            print(f"Error occurred while summarizing text: {e}")

    
    def summarize_audio(self, input: str) -> str:
        try:
            print('transcribing audio...')
            chunks, info = self.whisper.transcribe(input, best_of=1, beam_size=1, vad_filter=True)
            print('audio transcription complete')

            segments = list(chunks)

            print(f'processing {len(segments)} segments...')
            
            full_transcript = " ".join(seg.text for seg in segments)

            if full_transcript.strip():
                summary = self.summarize_text(full_transcript)
                print('Summary generated.')
                return summary
            else:
                print('No text extracted from audio.')
                return ""

        except Exception as e:
            print(f"Error occurred while summarizing audio: {e}")