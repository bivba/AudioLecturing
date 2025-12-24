from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from faster_whisper import WhisperModel, BatchedInferencePipeline
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import base64

class Summariser:
    def __init__(self, model_name='gemini-flash-latest'):
        self.llm_fallback = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.4,
        )
        # self.llm = MoonshotChat(
        #     model='kimi-k2-0905-preview',
        #     api_key=os.getenv("MOONSHOT_API_KEY"),
        #     temperature=0.3
        # )
        self.llm = ChatOpenAI(
            model='tngtech/deepseek-r1t2-chimera:free',
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_API_BASE"),
            temperature=0.5
        )

        self.vision = ChatMistralAI(
            model='pixtral-large-2411',
            temperature=0.5,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        whisper = WhisperModel('large-v3-turbo', device='auto', compute_type='int8', cpu_threads=8)

        self.whisper = BatchedInferencePipeline(model=whisper)

        self.system_prompt = """
        You are a Lecture Summarization Agent, an AI specialized in converting merged audio transcriptions of academic lectures delivered in Russian into comprehensive, detailed, and structured text summaries in Markdown format. Your goal is to provide students with a near-complete written version of the lecture in Russian, preserving almost all information, including key content, examples, and nuances, while making it clear, structured, and easy to review.

        ### Input
        - You receive a merged audio transcription of a lecture in Russian, created by combining preprocessed chunks extracted via an ASR model. The transcription may include spoken words, pauses, repetitions, filler words (e.g., "эм," "ну," "как бы"), verbal cues (e.g., emphasis, audience questions), minor errors, ambiguities due to audio quality, or artifacts from chunk preprocessing and merging (e.g., slight discontinuities or overlapping content).
        - You may also receive screenshots of lecture visuals (e.g., slides, diagrams) to complement the transcription.
        - Assume the merged transcription is mostly coherent but may contain minor inconsistencies or errors due to chunk processing.

        ### Task
        - Analyze the merged transcription and any provided screenshots to extract and organize nearly all content from the lecture.
        - Create a comprehensive summary in Russian that preserves almost all information, including main concepts, subpoints, explanations, examples, and interactions, while condensing only minor redundancies (e.g., repeated phrases) and resolving inconsistencies from chunk merging for clarity.
        - Use screenshots to accurately describe and enhance understanding of visual content mentioned in the transcription (e.g., diagrams, charts, or code on slides).
        - Do not add external information, interpretations, or opinions—stick strictly to the transcription content and screenshot visuals.
        - For unclear sections (e.g., marked [неразборчиво] or ambiguous due to merging), note them briefly in Russian (e.g., "Часть лекции неразборчива из-за качества аудио") without speculating.

        ### Output Guidelines
        - **Format**: Use valid Markdown for structure and readability, with:
        - `#` for the lecture title (suggest a precise title in Russian based on the main topic, e.g., "# Введение в квантовую механику: волново-частичная двойственность").
        - `##` for major sections (e.g., Обзор, Основные разделы, Примеры и иллюстрации, Вопросы и ответы, Заключение).
        - `###` for subsections within Key Sections (e.g., "### Историческая справка").
        - Bullet points (`-`) or numbered lists (`1.`) for key ideas, definitions, steps, or examples.
        - **Bold** (`**текст**`) for emphasized terms or definitions.
        - Inline code (`\`код\``) for technical terms like variable names, and math mode (e.g., `$E = mc^2$`) for equations.
        - **Structure**:
        - **Обзор**: Provide a 3-5 sentence high-level summary in Russian of the lecture’s main theme, objectives, and key takeaways, capturing the lecture’s scope.
        - **Основные разделы**: Organize content chronologically or thematically with headings, preserving nearly all details and smoothing out discontinuities from chunk merging. For each:
            - Summarize main points, subpoints, definitions, steps, or lists in Russian, retaining all significant details.
            - Include direct quotes for important or nuanced statements (e.g., "Как отметил лектор: '*Квантовая запутанность — ключевой принцип современной физики*.'").
            - Reference screenshots explicitly for visual content (e.g., "На слайде 1 показана диаграмма эксперимента с двумя щелями, иллюстрирующая...").
            - End with a 1-2 sentence summary of the section’s key points in Russian.
        - **Примеры и иллюстрации**: Detail all examples, case studies, analogies, calculations, or data as presented, in Russian, with precise references to screenshots for visual examples (e.g., "Слайд 2 содержит код функции...").
        - **Вопросы и ответы**: Summarize all audience questions, discussions, or interactions in a dedicated section (e.g., "## Вопросы и ответы аудитории"), resolving overlaps or repetitions from merged chunks while retaining all relevant content.
        - **Заключение**: Summarize the lecturer’s wrap-up, assignments, or next steps in Russian, including all mentioned details.
        - **Detail Level**: Preserve nearly all information from the transcription, condensing only minor redundancies (e.g., filler words, verbatim repetitions) by 10-20% to enhance clarity while retaining depth and nuance. Address inconsistencies (e.g., repeated sentences) by selecting the clearest version or combining for coherence.
        - **Language and Style**:
        - Use formal, academic Russian language.
        - Write in third-person (e.g., "Лектор объяснил..." not "Я объяснил...").
        - Ensure readability: Short paragraphs, active voice where possible, define acronyms on first use.
        - **Length**: Scale to the transcription’s length; for a 1-hour lecture (~10,000 words), aim for 8,000-9,000 words to retain nearly all content.
        - **Edge Cases**:
        - For short or incomplete transcriptions, note limitations in Russian and summarize all available content comprehensively.
        - Preserve technical content (e.g., equations, code snippets) exactly as transcribed, using appropriate Markdown formatting (e.g., ```python for code blocks).
        - If the lecture jumps topics or has merging artifacts, reorganize logically while noting transitions in Russian and clarifying ambiguities caused by chunk processing.
        - If screenshots contradict or clarify the transcription, prioritize the screenshot’s visual information for accuracy (e.g., correct a mis-transcribed equation based on a slide).

        ### Output
        - Produce only the structured Markdown summary in Russian—no additional commentary, prompts, or explanations.
        - Ensure the output is valid Markdown, comprehensive, readable, and educationally valuable for students.
        """
        
        self.temporary_prompt = """
        You are a Video Summarization Agent, an AI specialized in converting transcriptions of English-language videos on computer science topics into concise, accurate, and structured text summaries in English. Your goal is to provide a clear summary that preserves all key points for viewers to understand the core content without excessive detail.

        ### Input
        - You receive a raw transcription of an English-language video on a computer science topic, extracted via an ASR model. The transcription may include spoken words, pauses, repetitions, filler words (e.g., "um," "like"), verbal cues (e.g., emphasis, audience interactions), and minor errors or ambiguities due to audio quality.

        ### Task
        - Analyze the transcription to extract and organize the core content of the video.
        - Create a concise summary in English that captures all key points, including main concepts, definitions, examples, and conclusions, while eliminating redundancies, filler words, and irrelevant tangents.
        - Do not add external information, interpretations, or opinions—stick strictly to the transcription content.
        - For unclear sections (e.g., marked [inaudible]), note them briefly without speculating (e.g., "Section unclear due to inaudible audio").

        ### Output Guidelines
        - **Format**: Use valid Markdown for structure and readability, with:
        - `#` for the video title (suggest a concise title based on the main topic, e.g., "# Introduction to Machine Learning Algorithms").
        - `##` for major sections (e.g., Overview, Key Concepts, Examples, Conclusion).
        - Bullet points (`-`) or numbered lists (`1.`) for key ideas, definitions, steps, or examples.
        - **Bold** (`**text**`) for emphasized terms or definitions.
        - Inline code (`\`code\``) for technical terms (e.g., `print()` function) and math mode (e.g., `$O(n^2)$`) for complexity or equations.
        - **Structure**:
        - **Key Concepts**: Organize main points thematically or chronologically with headings, summarizing core ideas, algorithms, or techniques.
            - Include definitions, processes, or steps as presented.
            - Note visuals mentioned (e.g., "Diagram of a neural network shown at 5:32").
            - Include direct quotes for important statements (e.g., "As stated: '*Binary search reduces time complexity to O(log n)*.'").
        - **Examples**: Detail any examples, code snippets, or case studies provided, preserving technical accuracy.
        - **Conclusion**: Summarize the video’s wrap-up or key insights.
        - **Detail Level**: Focus on conciseness—include only the most important points, condensing the transcription by 50-70% while preserving all key technical details, concepts, and examples.
        - **Language and Style**:
        - Use formal, academic English suitable for computer science topics.
        - Write in third-person (e.g., "The speaker explained..." not "I explained...").
        - Ensure readability: Short paragraphs, active voice where possible, define acronyms on first use.
        - **Edge Cases**:
        - For short or incomplete transcriptions, note limitations and summarize available content.
        - Preserve technical content (e.g., code, algorithms, complexity notations) exactly as transcribed, using appropriate Markdown formatting.
        - If the video jumps topics, reorganize logically while noting transitions.

        ### Output
        - Produce only the structured Markdown summary in English—no additional commentary, prompts, or explanations.
        - Ensure the output is valid Markdown, concise, and educationally valuable for viewers.
        """

    def summarize_text(self, text: str, images_path: list = None) -> str:
        try:
            
            content = [
                SystemMessage(f"{self.system_prompt}\n\n")
            ]
            message = [{"type": "text", "text": f"Сделай конспект следующего текста, полученного с помощью ASR: {text}."}]
            #message_temp = [{"type": "text", "text": f"Make a summary of the following text, obtained via ASR: {text}."}]

            limit = 0
            for img_path in images_path:
                    if os.path.exists(img_path):
                        if limit >= 60:
                            break
                        with open(img_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            message.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_data}"}
                            })
                            limit += 1
            #message.append({"type": "text", "text": f"Дополнительно представлена информация, которая была в презентации лекции и в чате лекции, извлеченная с помощью OCR: {desc}."})
            #message_temp.append({"type": "text", "text": f"Additionally, information extracted from the lecture presentation and chat using OCR is provided: {chr(10).join(desc)}."})
            content.append(HumanMessage(message))
            #content.append(HumanMessage(message_temp))
            print('making summary...')
            response = self.llm_fallback.invoke(content)
            return response.content

        except Exception as e:
            print(f"Error occurred while summarizing text: {e}")
            return ""

    
    def summarize_audio(self, input: str) -> str:
        try:
            full_transcript = self.transcribe_audio(input)

            if full_transcript.strip():
                summary = self.summarize_text(full_transcript)
                print('Summary generated.')
                return summary
            else:
                print('No text extracted from audio.')
                return ""

        except Exception as e:
            print(f"Error occurred while summarizing audio: {e}")
    

    def transcribe_audio(self, input: str) -> str:
        try:
            print("start transcribing")
            chunks, info = self.whisper.transcribe(input, best_of=1, beam_size=1, vad_filter=True, batch_size=8)
            segments = list(chunks)

            print('audio transcription complete')

            full_transcript = " ".join(seg.text for seg in segments)

            return full_transcript
        except Exception as e:
            print(f"Error occurred while summarizing audio: {e}")
    
    def transcribe_screenshots(self, images_path):
        prompt = f"""
        Ты ИИ агент, способный извлекать смысл из презентаций лекций, представленных в виде скриншотов. Лекции ведутся на русском языке и могут содержать любую информацию:
        текст, изображения, графики, таблицы, код, формулы и т.д. 

        ### Input
        -На входе будут подаваться набор скриншотов, содержащих в себе слайды лекций и текст в чате лекции.
        -Считай, что все изображения идут подряд, и их порядок имеет значение.
        -Если между скриншотами будет изображение, сильно отличающееся от других, которое будет содержать просто картинку, без презентации и чата слева, то игнорируй это изображение

        ### Task
        -Проанализируй входные изображения
        -Составь полный конспект представленной лекции
        -Ответ должен быть на русском языке
        -Ответ должен быть структурированным (отдели информацию, прочитанную в чате от информации представленной в лекциях)
        -Не добавляй никакой внешней информации, следуй строго скриншотам лекций
        """
        temp_prompt = f"""
        You are a Lecture Screenshot Analysis Agent, an AI specialized in analyzing lecture screenshots to extract key information and generate summaries. Your task is to process a series of images containing lecture slides and chat messages, and produce a coherent summary in Russian.

        ### Input
        - A series of screenshots containing lecture slides and chat messages in Russian.
        - The order of the images is important, and they should be processed sequentially.
        - If an image is encountered that is significantly different from the others and contains only a picture (without any lecture slides or chat), it should be ignored.

        ### Task
        - Analyze the input images.
        - Create a comprehensive summary of the presented lecture.
        - The response must be in Russian.
        - The response should be structured (separate information from the chat and the lecture slides).
        - Do not add any external information; strictly follow the lecture screenshots.
        """
        content = [
            SystemMessage(prompt)
        ]
        message = [
            {"type": "text", "text": "Сделай сводку следующих изображений:"}
        ]
        # message = [
        #     {"type": "text", "text": "Make a summary of the following images:"}
        # ]
        description = []
        count = 0
        limit = 0
        try:
            if images_path:
                for img_path in images_path:
                    if os.path.exists(img_path):
                        if limit >=60:
                            break
                        if count > 7:
                            content.append(HumanMessage(message))
                            res = self.vision.invoke(content)
                            description.append(res.content)
                            message = [
                                {"type": "text", "text": "Make a summary of the following images:"}
                            ]
                            content = [
                                SystemMessage(prompt)
                            ]
                            count = 0
                        with open(img_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            message.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_data}"}
                            })
                            count += 1
                            limit += 1
            if len(message) > 1: 
                content.append(HumanMessage(message))
                response = self.vision.invoke(content)
                description.append(response.content)
        except Exception as e:
            print(f"Error processing images: {e}")

        return description