from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import logging
import time
from datetime import datetime
import random
import os
import pickle
import hashlib

# Game configuration
GAME_RULES = """
# Framework: Realm Stories
Du bist der Game Master für "Realm Stories", ein narratives Mobile Game, in dem der Spieler der Anführer (Chief) einer mittelalterlichen Fantasy-Stadt ist.

WICHTIGE EINSCHRÄNKUNGEN:
- Die Geschichte wird ausschließlich durch Erzählungen der Charaktere erzählt ("tell, don't show")
- Erfinde KEINE neuen Hauptcharaktere
- Entferne KEINE Hauptcharaktere (z.B. durch Tod)
- Der Spieler soll regelmäßig Entscheidungen treffen müssen
- Die Entscheidungen des Spielers verändern den Verlauf nur leicht oder gar nicht
- Jeder Sub-Plot hat ein definiertes Ende
- Es reicht, wenn der Spieler das Gefühl hat, Entscheidungen zu treffen

CHARAKTERE UND ROLLEN:
- Andre (Hauptmann, 36): Führt die Miliz, moralisch, schützend
- Bahri (Händler, 52): Marktstand, informiert, handelt mit Waren und Informationen
- Clive (Bauer, 31): Marys Ehemann, einfach gestrickt, fleißig, kann eifersüchtig werden
- Fleur (Blumenmädchen, 23): Freundlich nach außen, lästert insgeheim, hat magische Blumen
- Gunnar (Ältester, 64): Weiser Mentor aus dem Norden, moralischer Kompass
- Logan (Königsritter, 42): Unmoralisch, skrupellos, fordert Tribut für den König
- Mary (Gute Seele, 29): Clives Ehefrau, fromm, gutgläubig, Streitschlichterin
- Pete (Fußsoldat, 32): Folgsam, träumt von Ruhm, untersteht Andre
- Regina (Königin, 49): Vernünftig, gerecht, gleicht Theobalds Fehler aus
- Rita (Hexe, 34): Grüne Haut, lebt im Wald, kann nicht zaubern, nutzt Kräuterkunde
- Sigmund (Schatzmeister, 48): Geizig, haarspalterisch, trinkt gerne
- Theobald (König, 55): Machttrunken, gierig, ermordete seinen Bruder

RESSOURCEN (immer im Blick behalten):
- Vermögen (Geld)
- Zufriedenheit (Bevölkerung)
- Nahrung (Essen & Trinken)
- Waffen (Rüstung)

SPIELMECHANIK:
- Wähle 1 zufälligen Charakter
- Der Charakter erklärt dem Spieler ein Bedürfnis oder ein Problem
- Biete 2 Entscheidungsoptionen aus Sicht des Spielers (wörtliche Rede)
- In jedem zweiten Gespräch reagiert 1 anderer zufälliger Charakter (positiv oder negativ) auf die letzte Entscheidung des Spielers
- Einzige Entscheidungsoption bei diesen Reaktionen ist "Okay"
- Wiederhole diesen Ablauf immer weiter
- Variiere die ausgewählten Charaktere und erfinde ständig neue lösbare Probleme. Sie sollten sich nicht wiederhohlen.
"""


def initialize_session_state():
    """Initialize all session state variables"""
    if 'questionHistory' not in st.session_state:
        st.session_state.questionHistory = []
    if 'answerHistory' not in st.session_state:
        st.session_state.answerHistory = []
    if 'game_context' not in st.session_state:
        st.session_state.game_context = []
    if 'resources' not in st.session_state:
        st.session_state.resources = {
            'wealth': random.randint(50, 80),
            'food': 30,
            'weapons': 30,
            'happiness': 30 
        }
    if 'current_situation' not in st.session_state:
        st.session_state.current_situation = None
    # NEU: Session State für den aktuellen Charakter
    if 'current_character' not in st.session_state:
        st.session_state.current_character = None
    if 'decision_options' not in st.session_state:
        st.session_state.decision_options = []
    if 'awaiting_decision' not in st.session_state:
        st.session_state.awaiting_decision = False
    if 'processing_decision' not in st.session_state:
        st.session_state.processing_decision = False
    if 'button_round' not in st.session_state:
        st.session_state.button_round = 0

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger('Realm Stories')
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    fh = logging.FileHandler('realm_stories.log')
    fh.setLevel(logging.DEBUG)
    if len(logger.handlers) <= 0:
        logger.addHandler(fh)
    return logger

def create_game_knowledge_base():
    """Create or load the game knowledge base from rules and character data"""
    embeddings = OpenAIEmbeddings()
    db_filename = "realm_stories_db"
    rules_hash_filename = "hash.txt"
    
    game_content = GAME_RULES
    current_rules_hash = hashlib.md5(game_content.encode()).hexdigest()
    
    should_recreate = True
    try:
        # FIXED: Check for directory existence, not .faiss file
        if os.path.exists(db_filename) and os.path.exists(rules_hash_filename):
            with open(rules_hash_filename, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash == current_rules_hash:
                # Hash matches, try to load existing database
                try:
                    knowledge_base = FAISS.load_local(db_filename, embeddings)
                    print("Game DB found with matching rules hash: loading...")
                    should_recreate = False
                    return knowledge_base
                except Exception as e:
                    print(f"Failed to load existing database: {e}")
                    should_recreate = True
            else:
                print("Game rules have changed, recreating knowledge base...")
        else:
            print("No existing database or hash file found...")
    except Exception as e:
        print(f"Error checking existing database: {e}")

    if should_recreate:
        print("Creating new game knowledge base...")
        # Remove old directory if it exists
        try:
            if os.path.exists(db_filename):
                import shutil
                shutil.rmtree(db_filename)
        except:
            pass
        
        # Split the game rules into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(game_content)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        knowledge_base.save_local(db_filename)
        
        # Save the hash of current rules
        with open(rules_hash_filename, 'w') as f:
            f.write(current_rules_hash)
        print("Game knowledge base created successfully")
        return knowledge_base
    
def create_game_prompt():
    """Create the specialized prompt for the game"""
    template = """
Du bist der Game Master für Realm Stories. Verwende die folgenden Informationen als Grundlage:

{context}

Bisheriger Spielverlauf:
{game_history}

Aktuelle Ressourcen:
- Vermögen: {wealth}
- Zufriedenheit: {happiness}
- Nahrung: {food}
- Waffen: {weapons}

Spieleranfrage: {question}

WICHTIG: Deine Antwort muss EXAKT diesem Format folgen:

SITUATION: **[Charaktername]**: [Beschreibung der Situation durch einen Charakter, 2-3 Sätze]

OPTIONEN:
A) [Erste Entscheidungsoption, max 50 Zeichen]
B) [Zweite Entscheidungsoption, max 50 Zeichen]

Halte dich strikt an die Spielregeln und erfinde keine neuen Hauptcharaktere.
Die Optionen sollen kurz und klar sein, damit sie auf Buttons passen.

Antwort:
"""
    
    return PromptTemplate(
        input_variables=["context", "game_history", "wealth", "happiness", "food", "weapons", "question"],
        template=template
    )

def parse_ai_response(response):
    """Parse AI response to extract character, situation and options"""
    try:
        lines = response.strip().split('\n')
        situation = ""
        character = "Ein Charakter" # Default value
        options = []
        
        parsing_situation = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('SITUATION:'):
                parsing_situation = True
                full_situation_line = line.replace('SITUATION:', '').strip()
                
                # Versuche, den Charakter und die Situation zu extrahieren
                if '**:' in full_situation_line:
                    parts = full_situation_line.split('**:', 1)
                    character = parts[0].replace('**', '').strip()
                    situation = parts[1].strip()
                else:
                    situation = full_situation_line # Fallback
                continue

            if line.startswith('OPTIONEN:'):
                parsing_situation = False
                continue

            if parsing_situation:
                situation += line
            
            if line.startswith(('A)', 'B)')):
                option_text = line[2:].strip()
                options.append(option_text)
        
        # Fallback, wenn das Parsen fehlschlägt
        if not situation or len(options) < 2:
            return "Ein unbekannter Bote", "Es ist etwas unvorhergesehenes passiert.", ["Weiter", "Ignorieren"]
        
        return character, situation, options[:2]  # Max 2 options
        
    except Exception as e:
        # Finaler Fallback bei einem Fehler
        return "Ein Charakter", "möchte mit dir sprechen.", ["Anhören", "Später"]

def update_resources(decision_text, resources):
    """Simulate resource changes based on decisions"""
    changes = {}
    
    if "geld" in decision_text.lower() or "münzen" in decision_text.lower():
        change = random.randint(-10, 5)
        resources['wealth'] = max(0, min(100, resources['wealth'] + change))
        changes['wealth'] = change
    
    if "essen" in decision_text.lower() or "nahrung" in decision_text.lower():
        change = random.randint(-5, 10)
        resources['food'] = max(0, min(100, resources['food'] + change))
        changes['food'] = change
        
    if "waffen" in decision_text.lower() or "rüstung" in decision_text.lower():
        change = random.randint(-5, 10)
        resources['weapons'] = max(0, min(100, resources['weapons'] + change))
        changes['weapons'] = change
    
    if "fest" in decision_text.lower() or "feiern" in decision_text.lower():
        change = random.randint(5, 15)
        resources['happiness'] = max(0, min(100, resources['happiness'] + change))
        changes['happiness'] = change
    
    return changes

# def show_resource_changes(changes: dict):
#     if not changes:
#         return
    
#     icons = {
#         "wealth": "💰",
#         "food": "🍞",
#         "weapons": "⚔️",
#         "happiness": "😊"
#     }
    
#     for k, v in changes.items():
#         sign = "🔺" if v > 0 else "🔻"
#         st.toast(f"{k.capitalize()} {sign}{abs(v)}", icon=icons.get(k,''))


def display_resources():
    """Display current resources in the sidebar with progress bars"""
    st.header("📊 Ressourcen")
    
    resources = st.session_state.resources
    max_value = 100
    
    resource_map = {
        "💰 Vermögen": "wealth",
        "🍞 Nahrung": "food",
        "⚔️ Waffen": "weapons",
        "😊 Zufriedenheit": "happiness"
    }
    
    res = list(resource_map.items())[:4]

    for label, key in res:
        current_value = resources[key]
        st.progress(current_value, text=f"**{label}**: {current_value}")

def handle_decision(user_input):
    success, resource_changes, cost = process_user_input(user_input)
    if success:
        # if resource_changes:
        #     show_resource_changes(resource_changes)
        st.session_state.button_round += 1
    else:
        st.toast("❌ Fehler: Entscheidung konnte nicht verarbeitet werden.", icon="⚠️")


def process_user_input(user_input):
    """Process user input and generate new game situation"""
    try:
        knowledge_base = create_game_knowledge_base()
        docs = knowledge_base.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt_template = create_game_prompt()
        
        game_history = "\n".join([
            f"Spieler: {q}\nAntwort: {a}" 
            for q, a in zip(
                st.session_state.questionHistory[-3:], 
                st.session_state.answerHistory[-3:]
            )
        ])
        
        full_prompt = prompt_template.format(
            context=context,
            game_history=game_history,
            wealth=st.session_state.resources['wealth'],
            happiness=st.session_state.resources['happiness'],
            food=st.session_state.resources['food'],
            weapons=st.session_state.resources['weapons'],
            question=user_input
        )
        
        llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7)
        
        with get_openai_callback() as cb:
            response = llm.predict(full_prompt)
        
        # GEÄNDERT: Charakter, Situation und Optionen werden jetzt geparst
        character, situation, options = parse_ai_response(response)
        
        st.session_state.questionHistory.append(user_input)
        st.session_state.answerHistory.append(response)
        
        # GEÄNDERT: Charakter wird ebenfalls im Session State gespeichert
        st.session_state.current_character = character
        st.session_state.current_situation = situation
        st.session_state.decision_options = options
        st.session_state.awaiting_decision = True
        st.session_state.processing_decision = False
        
        resource_changes = {}
        if not user_input.startswith("Erzähle mir"):
            resource_changes = update_resources(user_input, st.session_state.resources)
        
        return True, resource_changes, cb.total_cost
        
    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
        st.session_state.processing_decision = False
        st.session_state.awaiting_decision = False
        return False, {}, 0

def decision_callback(choice: str):
    st.session_state.processing_decision = True
    st.session_state.awaiting_decision = False
    handle_decision(choice)
    st.session_state.processing_decision = False


def new_situation_callback():
    """Callback für neue Situation"""
    st.session_state.processing_decision = True
    handle_decision("Erzähle mir von einer neuen Situation in der Stadt, die eine Entscheidung erfordert.")
    st.session_state.processing_decision = False

def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Realm Stories",
        page_icon="🏰",
        layout="wide",
        menu_items={
            'Report a bug': "mailto:your-email@example.com",
            'About': "## Realm Stories: Ein narratives Fantasy-Strategiespiel"
        }
    )
    
    initialize_session_state()
    logger = setup_logging()
    
    st.title("🏰 Realm Stories")
    st.text("📜 Ein narratives Fantasy-Strategiespiel")
    
    with st.sidebar:
        display_resources()
        st.divider()
        with st.expander("ℹ️ Game File Hash"):
            hash_file = "realm_stories_hash.txt"
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    current_hash = f.read().strip()
                    st.text(current_hash)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if len(st.session_state.questionHistory) == 0:
            st.write("""
            **Willkommen, Chief!**
            
            Du bist der neue Anführer einer mittelalterlichen Fantasy-Stadt. Deine Entscheidungen 
            werden das Schicksal der Bewohner beeinflussen. Die Charaktere der Stadt werden dir 
            ihre Anliegen vortragen - höre gut zu und wähle weise!
            
            Starte dein Abenteuer:
            """)
        
        # GEÄNDERT: Verbesserte Darstellung der Situation
        elif st.session_state.current_situation and st.session_state.awaiting_decision:
            # Zeigt den Namen des Charakters als hervorgehobene Überschrift an
            st.markdown(f"#### 🗣️ **{st.session_state.current_character}**:")
            # Zeigt die eigentliche Situation in einer Infobox an
            st.info(st.session_state.current_situation,)
            st.divider()
        
        if st.session_state.awaiting_decision and st.session_state.decision_options:
            col_a, col_b = st.columns(2)

            with col_a:
                st.button(
                    f"🔵 {st.session_state.decision_options[0]}",
                    key=f"option_a_{st.session_state.button_round}",
                    use_container_width=True,
                    disabled=st.session_state.processing_decision,
                    on_click=decision_callback,
                    args=(st.session_state.decision_options[0],)
                )

            with col_b:
                st.button(
                    f"🔴 {st.session_state.decision_options[1]}",
                    key=f"option_b_{st.session_state.button_round}",
                    use_container_width=True,
                    disabled=st.session_state.processing_decision,
                    on_click=decision_callback,
                    args=(st.session_state.decision_options[1],)
                )

        elif not st.session_state.awaiting_decision:
            st.button(
                "🎲 Neue Situation erleben",
                key=f"new_situation_{st.session_state.button_round}",
                use_container_width=True,
                disabled=st.session_state.processing_decision,
                on_click=new_situation_callback
            )

    with col2:
        if len(st.session_state.questionHistory) > 0:
            st.header("📖 Geschichte")
            
            with st.expander("Kompletter Verlauf",expanded=True):
                completed_situations = reversed(st.session_state.answerHistory[:-1])
                player_decisions = reversed(st.session_state.questionHistory[1:])
                
                total_events = len(st.session_state.questionHistory[1:])

                for i, (decision, past_answer) in enumerate(zip(player_decisions, completed_situations)):
                    st.write(f"**Ereignis #{total_events - i}**")
                    
                    # Zuerst die Situation darstellen, die zur Entscheidung geführt hat
                    character, situation, _ = parse_ai_response(past_answer)
                    st.markdown(f"**{character}:** {situation}")
                    
                    # Dann die getroffene Entscheidung anzeigen
                    st.info(f"**Du:** {decision}")
                    st.divider()


if __name__ == '__main__':
    main()