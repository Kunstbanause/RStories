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

TEXTBEISPIELE:
Alle Zeilen, deren KEY identisch oder ähnlich ist, gehören zu einer SITUATION.
Nach dem KEY folgt, welcher CHARAKTER gerade spricht, und danach der gesprochene DIALOG.
Wenn CHARAKTER="Chief", dann ist das eine Gesprächsoption des Spielers.
Verstehe, welche Dialog-Zeilen zusammengehören (zur selben SITUATION) und wie die Interaktion zwischen dem CHARAKTER und dem Spieler funktioniert.

CSV-Format:
KEY,CHARAKTER,DIALOG
sigmund_buys_food_or_arms_1,Sigmund,"Nun, es scheint, wir verfügen über ein kleines Finanzpolster."
sigmund_buys_food_or_arms_2,Sigmund,Welchem Zweck sollen wir das Geld zuführen?
sigmund_buys_food_or_arms_food,Chief,Wir brauchen Getreide.
sigmund_buys_food_or_arms_arms,Chief,Wir brauchen Rüstungen.
sigmund_buys_food_1,Sigmund,"{0}, wir haben etwas Geld übrig. Nicht schlecht, was?"
sigmund_buys_food_2,Sigmund,Sollen wir es für frisches Gemüse investieren oder lieber sparen?
sigmund_buys_food_yes,Chief,"Ja, wir kaufen Gemüse."
sigmund_buys_food_no,Chief,"Nein, wir sparen für schlechte Zeiten."
sigmund_buys_arms_1,Sigmund,Unser Budget lässt Spielraum für Investitionen.
sigmund_buys_arms_2,Sigmund,Wollen wir in Waffen investieren oder lieber ansparen?
sigmund_buys_arms_yes,Chief,"Ja, wir kaufen Waffen."
sigmund_buys_arms_no,Chief,"Nein, Gold ist besser als Stahl."
sigmund_raises_tax_1,Sigmund,Die Kassen sind beunruhigend leer.
sigmund_raises_tax_2,Sigmund,Sollen wir vielleicht die Steuern anheben?
sigmund_raises_tax_yes,Chief,"Ja, wir brauchen mehr Geld."
sigmund_raises_tax_no,Chief,Nicht zur Last unserer Leute.
sigmund_sells_arms_1,Sigmund,"Unsere Finanzen sind dünn, {0}."
sigmund_sells_arms_2,Sigmund,Vielleicht sollten wir ein paar Schilde veräußern?
sigmund_sells_arms_yes,Chief,"Ja, wir sind stark genug gerüstet."
sigmund_sells_arms_no,Chief,"Nein, unsere Miliz braucht die Schilde."
andre_recruit_people_1,Andre,"{0}, ich denke, es ist an der Zeit, unsere Verteidigung zu stärken."
andre_recruit_people_2,Andre,"Wie wäre es, wenn wir junge Männer und Frauen rekrutieren, um unsere Reihen aufzufüllen?"
andre_recruit_people_yes,Chief,"Ja, rekrutiere Leute aus dem Volk."
andre_recruit_people_no,Chief,"Nein, die Miliz ist stark genug."
andre_hunt_start_1,Andre,Wir könnten ein paar meiner Leute tagsüber auf die Jagd schicken. Das Fleisch ist wichtig für alle hier.
andre_hunt_start_2,Andre,Sollen wir auf die Jagd gehen?
andre_hunt_start_yes,Chief,"Ja, die Leute brauchen was zum essen."
andre_hunt_start_no,Chief,"Nein, bewacht lieber die Stadt."
andre_hunt_result_great,Andre,"Du glaubst gar nicht, wie erfolgreich wir waren! Unsere Kammern sind voll mit Wildfleisch!"
andre_hunt_result_good,Andre,"Die Jagd war in Ordnung. Wir haben genug erlegt, um uns über die Runden zu bringen."
andre_hunt_result_bad,Andre,Es war ein harter Tag. Wir sind mit leeren Händen von der Jagd zurückgekommen.
mary_night_watch_1,Mary,"Ich mach' mir Sorgen um unser Städtchen bei Nacht. Es gab erst neulich wieder Umtriebe, und die Leute fühlen sich nicht sicher."
mary_night_watch_2,Mary,"Könntest du nicht ein paar Männer der Miliz einteilen, die dann nachts Wache halten? Nur damit alle beruhigt schlafen können."
mary_night_watch_yes,Chief,"Ja, wir verstärken die Nachtwache."
mary_night_watch_no,Chief,"Nein, die Männer müssen sich erholen."
mary_feast_1,Mary,Unsere Kammern sind randvoll mit Vorräten. Wie wäre es mit einem Fest?
mary_feast_2,Mary,Ein bisschen Freude würde allen gut tun. Wir sollten dankbar feiern!
mary_feast_yes,Chief,"Ja, wir veranstalten ein Fest."
mary_feast_no,Chief,"Nein, es gibt wichtigeres."
pete_night_watch_safe,Pete,"{0}, die Nacht war ruhig. Die Stadt ist sicher, kein Räuber weit und breit."
pete_night_watch_tired_1,Pete,"{0}, auf der Nachtwache letzte Nacht, da war dieses seltsame Geräusch."
pete_night_watch_tired_2,Pete,"War schon spät, ich hab praktisch auf den Beinen geschlafen. Wahrscheinlich war's nichts."
pete_wants_beer_1,Pete,Wir von der Miliz könnten nach all der harten Arbeit etwas zum Runterkommen gebrauchen.
pete_wants_beer_2,Pete,"Wie wär's, du lässt uns ein paar Fässer Bier zukommen, als Lohn für unsere Treue?"
pete_wants_beer_yes,Chief,"Ja, das habt ihr euch verdient."
pete_wants_beer_no,Chief,"Nein, ihr sollt nüchtern bleiben."
clive_harvest_good,Clive,Ich habe reichlich Gemüse geerntet. Wir sind gut versorgt!
clive_harvest_bad,Clive,"Die Ernte ist mies ausgefallen, zu wenig Regen, und die Felder sind dürr. Wir müssen aufpassen."
bahri_trade_good,Bahri,"{0}, die Geschäfte brummen. Meine Ware geht weg wie warme Semmeln."
bahri_trade_bad,Bahri,"Momentan sieht's düster aus, die Kunden bleiben weg. Hart, aber ich bleib dran."
bahri_sentiment_good,Bahri,"Die Stimmung ist bestens, {0}. Überall lachende Gesichter und optimistische Gespräche."
bahri_sentiment_bad,Bahri,Die Leute sind unruhig und murren. Es brodelt in den Gassen.
fleur_loves_militia,Fleur,"Es ist einfach so... erhebend zu sehen, wie stark unsere Miliz momentan ist. Da ist etwas an der Präsenz so vieler starker Männer, das mich ganz... besonders fühlen lässt."
fleur_wants_food,Fleur,"Ich kann es kaum fassen, schon wieder nur trockenes Brot! Es ist zum Haare raufen – wann hatten wir das letzte Mal etwas richtig Gutes zu essen?"
logan_demands_wheat_or_bribe_1,Logan,"Ich bin hier wegen eures Getreides. Aber weißt du, man kann über alles reden."
logan_demands_wheat_or_bribe_2,Logan,"Zwischen dir und mir, es gibt Wege, wie wir beide davon profitieren können."
logan_demands_wheat_or_bribe_wheat,Chief,Hier ist euer Getreide.
logan_demands_wheat_or_bribe_bribe,Chief,Nehmt stattdessen etwas Geld.
logan_demands_food_sigmund_reaction,Sigmund,Logan zu bestechen war ein Fehler. Wir graben uns unser eigenes finanzielles Grab.
logan_demands_food_clive_reaction,Clive,"Es beunruhigt mich, dass du Logan das Getreide überlassen hast. Wenn das so weitergeht, müssen unsere Leute hungern..."
logan_demands_wealth_1,Logan,"Zeit, den Tribut für den König einzusammeln. Das ist kein Besuch, den jemand gerne hat."
logan_demands_wealth_2,Logan,"Ich empfehle dir, direkt zu zahlen. Dann musst du mich kein zweites Mal sehen."
logan_demands_wealth_agree,Chief,Wir zahlen den Tribut.
logan_demands_wealth_bribe,Chief,Trinkt lieber von unserem Wein.
logan_demands_wealth_bribe_gunnar_relief_1,Gunnar,"Wir dürfen aufatmen, denn unsere Gastfreundschaft hat Logan und seine Männer zufriedengestellt. Doch die Erleichterung währt kurz."
logan_demands_wealth_bribe_gunnar_relief_2,Gunnar,"Bei aller Gerissenheit frage ich mich, ob Wein statt Tribut auch beim nächsten Mal ihre Gier stillen kann."
logan_demands_wealth_bribe_gunnar_revelry_1,Gunnar,"Was als einfaches Gelage begann, hat sich zum totalen Chaos entwickelt. Logan und seine Truppe haben die Stadt auf den Kopf gestellt."
logan_demands_wealth_bribe_gunnar_revelry_2,Gunnar,"Überall verängstigte Gesichter. Ihre Feierei hat nicht nur Lärm und Unordnung gebracht, sondern auch Angst unter den Bewohnern."
logan_demands_wealth_bribe_gunnar_revelry_3,Gunnar,"Hättest du das kommen sehen müssen? Du hast sie mit Wein willkommen geheißen, aber vielleicht war das ein Fehler."
logan_demands_wealth_agree_sigmund_reaction_1,Sigmund,Das direkte Zahlen des Tributs an Logan passt mir gar nicht in den Kram. Jedes Goldstück ist wichtig.
logan_demands_wealth_agree_sigmund_reaction_2,Sigmund,"Ich bitte dich, denk das nächste Mal um die Ecke. Es muss doch Möglichkeiten geben, ohne direkte Zahlung auszukommen."
logan_demands_arms_1,Logan,"Die Zeiten verlangen nach starken Maßnahmen. Ich suche bewaffnete Männer, die im Namen des Königs kämpfen können."
logan_demands_arms_2,Logan,Eure Kämpfer sind jetzt gefragt. Übergebt sie dem Dienst des Königs.
logan_demands_arms_agree,Chief,Okay aber behandelt sie ordentlich.
logan_demands_arms_bribe,Chief,Nehmt etwas Geld und lasst uns die Männer.
logan_demands_arms_bribe_andre_reaction_1,Andre,"{0}, ich kann gar nicht genug danken. Das Geld, das du Logan gegeben hast, sagt mehr als Worte."
logan_demands_arms_bribe_andre_reaction_2,Andre,"Du hast unsere Männer vor dem Schicksal in der königlichen Armee bewahrt. Das ist kein leichtes Leben, glaub mir."
logan_demands_arms_bribe_andre_reaction_3,Andre,"Ich weiß, was es heißt, da draußen zu kämpfen. Du hast uns allen einen großen Dienst erwiesen."
logan_demands_arms_agree_pete_reaction_1,Pete,"Ich hab's gesehen, wie sie ein paar von uns mitgenommen haben. Ist das unser Schicksal jetzt, einfach ausgewählt zu werden?"
logan_demands_arms_agree_pete_reaction_2,Pete,"Das ist beängstigend, weißt du? Ich will kämpfen, ja. Aber nicht so, nicht gezwungen und fern von zu Hause."
logan_demands_arms_agree_pete_reaction_3,Pete,"Was, wenn ich der Nächste bin? Mann, ich bin nicht bereit, mein Glück so aufs Spiel zu setzen."
fleur_complains_about_taxes_1,Fleur,"Oh, da bist du ja! Ich muss wirklich mal loswerden, was für ein Ärgernis das wieder ist. Die Steuererhöhung kommt ja so überraschend wie Regen im Frühling."
fleur_complains_about_taxes_2,Fleur,"Und wer steckt dahinter? Sigmund natürlich! Könnte ja mal anfangen, weniger zu prassen, statt uns ständig mehr abzuverlangen."
fleur_complains_about_taxes_3,Fleur,"Ich schwör, eines Tages... Ach, vergiss es. Solange wir zusammenhalten, wird’s schon gehen."
sarebia_theobald_proud_1,Theobald,Die Grenzüberwachung bei Sarebia verläuft vorbildlich. Unsere Männer dort machen mich stolz.
sarebia_theobald_proud_2,Theobald,"{0}, du solltest auch stolz sein. Ihre Tapferkeit verdient Anerkennung."
sarebia_logan_boasts_1,Logan,Unsere Truppen machen fantastische Fortschritte an der Grenze zu Sarebia. Wieder einmal haben wir unsere Stärke unter Beweis gestellt.
sarebia_logan_boasts_2,Logan,"Es gab Kämpfe, bei denen die königliche Armee ihre klare Überlegenheit gezeigt hat. Ein deutliches Zeichen unserer Macht."
sarebia_theobald_complains_1,Theobald,Die Berichte über deine Männer an der Grenze beunruhigen mich. Sie wirken... verweichlicht.
sarebia_theobald_complains_2,Theobald,"Ich frage mich, ob du wirklich an unsere Sache glaubst, {0}. Ziehen wir wirklich am selben Strang?"
sarebia_regina_requests_wealth_1,Regina,Die Menschen in Sarebia leiden unter der harten Hand unserer königlichen Armee. Ihre Notlage ist tiefgreifend und erfordert unsere Aufmerksamkeit.
sarebia_regina_requests_wealth_2,Regina,"Deshalb bitte ich dich, {0}, eine Lieferung von medizinischen Tinkturen und Verbänden dorthin zu schicken."
sarebia_regina_requests_wealth_3,Regina,Deine Hilfe kann viele Leben retten.
sarebia_regina_requests_wealth_yes,Chief,"Ja, wir müssen diesen Leuten helfen."
sarebia_regina_requests_wealth_no,Chief,"Nein, das können wir uns nicht leisten."
sarebia_regina_requests_arms_1,Regina,"Die königlichen Truppen haben erneut ein Dorf in Sarebia verwüstet. Die Nachricht bricht mir das Herz. Wir müssen zeigen, dass nicht alle von uns Feinde sind."
sarebia_regina_requests_arms_2,Regina,"Ich bitte dich inständig, {0}, deinen Soldaten zu befehlen, bei den Reparaturen zu helfen."
sarebia_regina_requests_arms_yes,Chief,"Ja, unsere Männer packen mit an."
sarebia_regina_requests_arms_no,Chief,"Nein, die Truppe ist erschöpft."
sarebia_regina_requests_food_1,Regina,"Es schmerzt mich zutiefst, dass unsere Truppen weiterhin die sarebischen Dörfer überfallen und ihnen essentielle Nahrungsmittel entwenden."
sarebia_regina_requests_food_2,Regina,"Ich appelliere an dich, {0}, eine Lieferung mit Früchten und Brot zu den betroffenen Dörfern in Sarebia zu arrangieren."
sarebia_regina_requests_food_yes,Chief,"Ja, wir schicken ihnen Vorräte"
sarebia_regina_requests_food_no,Chief,"Nein, sonst müssen wir selber hungern."
sarebia_andre_reports_losses_1,Andre,"Es gibt traurige Nachrichten von der Grenze zu Sarebia. Die Situation hat sich zugespitzt, und es kam zu einem Kampf."
sarebia_andre_reports_losses_2,Andre,Leider mussten wir Verluste hinnehmen. Einige unserer Männer sind gefallen. Es ist ein schwerer Schlag für uns alle.
sarebia_bahri_is_happy_1,Bahri,"Ich habe von meinen Freunden gehört, dass sich die Lage in Sarebia beruhigt hat. Das ist eine gute Nachricht."
sarebia_bahri_is_happy_2,Bahri,"Für deine Unterstützung dort, {0}, bin ich dir unendlich dankbar."
sarebia_sigmund_dislikes_help_1,Sigmund,"Dass du in diesen Konflikt mit Sarebia eingreifst, finde ich völlig unangebracht. Wir sollten unseren eigenen Wohlstand im Blick haben."
sarebia_sigmund_dislikes_help_2,Sigmund,"Den Reichtum unserer Stadt für diese Fremden zu verwenden, entbehrt jeglicher Vernunft. Wir haben hier genug eigene Probleme."
pete_complains_about_selling_arms_1,Pete,"{0}, wie soll das nur weitergehen, ohne unsere Schilde?"
pete_complains_about_selling_arms_2,Pete,"Ich versteh schon, dass Geld wichtig ist, aber was, wenn es mal ernst wird?"
pete_complains_about_selling_arms_3,Pete,"Banditen, Monster, sogar Drachen könnten uns angreifen! Da steht man ohne Schild dumm da."
pete_complains_about_selling_arms_4,Pete,"Ich will ja nicht nörgeln, aber ohne Schild fühle ich mich nackt. Nicht mal richtig wehren kann man sich."
andre_thanks_for_buying_arms_1,Andre,Ein großes Dankeschön von der Miliz für deine nicht enden wollende Unterstützung.
andre_thanks_for_buying_arms_2,Andre,"Indem du wiederholt in unsere Ausrüstung investierst, zeigst du tiefes Engagement für unsere Sicherheit."
andre_thanks_for_buying_arms_3,Andre,"Dank dir sind wir stets gut gerüstet, um unsere Stadt und ihre Bewohner zu verteidigen. Deine Fürsorge ist wirklich unbezahlbar."
mary_thanks_for_buying_food_1,Mary,"{0}, vom tiefsten Grunde meines Herzens, danke ich dir für deine Gnade und Fürsorglichkeit."
mary_thanks_for_buying_food_2,Mary,"Dass du immer wieder dafür sorgst, dass unsere Speicher gefüllt sind, lässt mich ruhig schlafen. Niemand muss hungern."
mary_thanks_for_buying_food_3,Mary,Du bist wahrlich ein Segen für uns alle. Gott segne dich für deine Großzügigkeit und Liebe zu unserer Gemeinschaft.
gunnar_complains_about_drunk_militia_1,Gunnar,"{0}, ich muss dir eine Sache ans Herz legen, die mir Sorgen bereitet."
gunnar_complains_about_drunk_militia_2,Gunnar,"Das viele Bier für unsere Miliz... es macht sie träge und unachtsam, wenn es darauf ankommt."
gunnar_complains_about_drunk_militia_3,Gunnar,"Bedenke stets, dass ein klarer Geist und wache Sinne im Einsatz über Leben und Tod entscheiden können."
gunnar_praises_sober_militia_1,Gunnar,"Ich muss sagen, ich bewundere deine Strenge bezüglich des Biers."
gunnar_praises_sober_militia_2,Gunnar,"Sicher, die Jungs von der Miliz murren jetzt, aber am Ende des Tages hältst du sie fit und bereit."
gunnar_praises_sober_militia_3,Gunnar,"Was wirklich zählt, ist, dass sie scharf und zu jeder Zeit einsatzbereit bleiben. Gut gemacht!"
mary_complains_about_recruiting_people_1,Mary,"{0}, ich mache mir Sorgen. Du holst immer öfter junge Leute aus dem Dorf in die Miliz."
mary_complains_about_recruiting_people_2,Mary,"Nicht jeder träumt davon, zu kämpfen. Einige haben Angst, und ich verstehe sie."
mary_complains_about_recruiting_people_3,Mary,"Als Mutter fürchte ich den Tag, an dem meine eigenen Kinder alt genug sind und du sie rufst."
mary_complains_about_recruiting_people_4,Mary,"Ich bete, dass es nie soweit kommt. Gewalt sollte immer der letzte Ausweg sein."
gunnar_refugees_men_1,Gunnar,"Draußen vor den Toren, wurden wir von einer Gruppe junger Männer angesprochen. Sie suchen Schutz und ein Dach über dem Kopf."
gunnar_refugees_men_2,Gunnar,"Es obliegt deiner Weisheit, über das Schicksal dieser Seelen zu entscheiden. Sollen wir ihnen den Zutritt gewähren?"
gunnar_refugees_men_yes,Chief,"Ja, wir lassen die Männer herein."
gunnar_refugees_men_no,Chief,"Nein, wir schicken sie weg."
gunnar_refugees_women_1,Gunnar,Vor den Toren stehen zwei Frauen mittleren Alters. Sie halten einander fest und suchen Schutz. Ihr Anblick berührt mich sehr.
gunnar_refugees_women_2,Gunnar,"Sollen wir sie aufnehmen? Ich denke, sie könnten unsere Gemeinschaft bereichern."
gunnar_refugees_women_yes,Chief,"Ja, wir lassen die Frauen herein."
gunnar_refugees_women_no,Chief,"Nein, wir schicken sie weg."
gunnar_refugees_impaired_man_1,Gunnar,"Ein bärtiger Mann mit nur einem Auge und einem lahmen Bein steht vor den Toren. Er hat mir erzählt, dass er sein Zuhause verloren hat."
gunnar_refugees_impaired_man_2,Gunnar,Sollen wir ihn aufnehmen? Er könnte vielleicht eine zweite Chance gebrauchen.
gunnar_refugees_impaired_man_yes,Chief,"Ja, nehmen wir den Mann auf."
gunnar_refugees_impaired_man_no,Chief,"Nein, er wird uns nur zur Last fallen."
gunnar_refugees_family_1,Gunnar,"Eine Familie mit Kindern bittet vor den Toren um Zuflucht. Sie sehen mitgenommen aus, aber hoffnungsvoll."
gunnar_refugees_family_2,Gunnar,Soll ich sie hereinlassen? Das liegt in deinen Händen.
gunnar_refugees_family_yes,Chief,"Ja, wir nehmen sie auf."
gunnar_refugees_family_no,Chief,"Nein, sie sollen weiterziehen."
gunnar_refugees_oldlady_1,Gunnar,"Vor den Toren steht eine alte Dame, ganz allein, und sie bittet um Unterschlupf. Ihre Augen erzählen Geschichten, {0}."
gunnar_refugees_oldlady_2,Gunnar,"Sollen wir sie aufnehmen? Was du entscheidest, prägt das Herz unserer Gemeinschaft."
gunnar_refugees_oldlady_yes,Chief,"Ja, sie soll hereinkommen."
gunnar_refugees_oldlady_no,Chief,"Nein, sie soll draußen bleiben."
sigmund_praises_refugee_workers_1,Sigmund,"Ich muss zugeben, einige dieser Fremden sind echt fleißig. Sie packen kräftig mit an und das täglich von Morgens bis Abends."
sigmund_praises_refugee_workers_2,Sigmund,"Sie sind zwar nicht von hier, aber sie erledigen ihre Arbeit gut und vor allem sind sie eine günstige Hilfe. Das kann man nicht abstreiten."
mary_praises_refugee_kids_1,Mary,"Es ist wirklich herzerwärmend zu sehen, wie meine Kleinen mit den neuen Kindern auskommen. Sie spielen den ganzen Tag zusammen, und es scheint, als hätten sie nie etwas anderes getan."
mary_praises_refugee_kids_2,Mary,"Ich war anfangs besorgt, wie sie aufeinander reagieren würden, aber Kinder haben so eine reine und annehmende Art. Sie sehen keine Unterschiede, nur Spielkameraden."
mary_praises_refugee_kids_3,Mary,"Es beruhigt mich und gibt mir Hoffnung, dass unsere Gemeinschaft nur stärker wird, wenn wir offen und einladend sind. Danke, {0}, dass du diese Familien hierher gebracht hast."
andre_praises_accepting_refugees_1,Andre,"Du wirst es kaum glauben, aber unter den Neuankömmlingen haben wir echte Kämpfernaturen entdeckt. Das ist ein Glücksfall für uns."
andre_praises_accepting_refugees_2,Andre,Ihre Fähigkeiten und ihr Mut sind beeindruckend. Sie bringen eine frische Brise und neue Stärke in unsere Truppe ein.
andre_praises_accepting_refugees_3,Andre,"Das hebt die Moral meiner Männer. Wir sind nicht nur stärker geworden, sondern auch vereinter. Es war die richtige Entscheidung, sie aufzunehmen."
bahri_praises_accepting_refugees_1,Bahri,"Als jemand, dessen Wurzeln in fernen Ländern liegen, weiß ich die Geste, Fremden ein Zuhause zu bieten, besonders zu schätzen."
bahri_praises_accepting_refugees_2,Bahri,"Meiner Familie wurde hier einst Zuflucht gewährt, und zu sehen, wie du diese Tradition der Offenheit und Gastfreundlichkeit fortführst, erfüllt mich mit Stolz."
bahri_praises_accepting_refugees_3,Bahri,"Du zeigst wahren Mut und eine seltene Güte, die in diesen Zeiten mehr denn je gebraucht wird. Es ist eine Ehre, dir zu dienen."
sigmund_complains_about_refugees_1,Sigmund,"Du hast zweifelsohne ein großes Herz, doch ich fürchte, es ist zu weich für diese Welt. Menschen strömen in Scharen zu uns, von überall her."
sigmund_complains_about_refugees_2,Sigmund,"Man muss realistisch bleiben. Unsere Ressourcen sind begrenzt, und diese ständige Gastfreundschaft könnte uns in den Ruin treiben."
sigmund_complains_about_refugees_3,Sigmund,"Wir müssen dringend überdenken, wem wir unsere Türen öffnen. Sonst zahlen wir alle den Preis für ein zu gütiges Herz."
clive_complains_about_refugees_1,Clive,"Ich finde es ja nett, dass du den fremden Leuten ein neues Zuhause gibt, echt jetzt."
clive_complains_about_refugees_2,Clive,"Aber wir haben jetzt noch mehr Mäuler zu stopfen, mehr als vorher schon. Das macht mir Sorgen."
clive_complains_about_refugees_3,Clive,"Ich hoffe nur, wir kriegen das alles gebacken. Sonst wird's düster für uns alle, nich'?"
gunnar_reports_happy_refugees_1,Gunnar,"Die Leute, die du hierher geholt hast, haben sich echt gut eingelebt. Sie sind über alle Maßen dankbar für die Chance, die du ihnen gegeben hast."
gunnar_reports_happy_refugees_2,Gunnar,"Sie bringen so viel Neues mit – Künste, Wissen, Geschichten. Unsere Stadt fühlt sich reicher an, lebendiger irgendwie."
gunnar_reports_happy_refugees_3,Gunnar,"Zu sehen, wie sie sich hier einleben, macht mich stolz. Sie sind ein Beweis dafür, dass deine Entscheidung mehr als richtig war."
rita_complains_about_hunt_1,Rita,"Ich bin echt sauer, dass du immer wieder diese Männer in den Wald schickst, um zu jagen."
rita_complains_about_hunt_2,Rita,"Sie wissen doch gar nicht, was sie da anrichten. Der Wald und seine Tiere leiden."
rita_complains_about_hunt_3,Rita,"Mir bricht es fast das Herz, wenn ich nur dran denke, wie die Tiere gehetzt und getötet werden."
rita_complains_about_hunt_4,Rita,"Denk doch bitte noch einmal darüber nach. Der Wald und ich, wir wären dir dankbar."
rita_praises_no_more_hunt_1,Rita,"Ich wollte dir was Schönes sagen. Der Wald blüht richtig auf, seitdem niemand mehr jagt."
rita_praises_no_more_hunt_2,Rita,"Die Tiere wirken viel entspannter. Es ist, als ob die ganze Gegend aufatmet."
rita_praises_no_more_hunt_3,Rita,"Du kannst dir kaum vorstellen, was das für mich bedeutet. Das macht mich richtig glücklich!"
rita_praises_no_more_hunt_4,Rita,"Hoffentlich bleibt es dabei. Es ist ein Geschenk, den Wald so lebendig zu sehen."
rita_gifts_herbs_1,Rita,"Du, ich hab in meinem Garten ein bisschen was über. Viel Rosmarin und Thymian."
rita_gifts_herbs_2,Rita,"Echt, ich hab mehr davon, als ich je verwenden könnte. Dachte, ich teile mal ein bisschen."
rita_gifts_herbs_3,Rita,Ich würde die Kräuter gerne der Stadt schenken. Hättest du Interesse daran?
rita_gifts_herbs_yes,Chief,"Ja, klar!"
rita_gifts_herbs_no,Chief,"Nein, danke."
andre_praises_herbs_1,Andre,Hab ich das richtig geschmeckt? Das Essen ist in letzter Zeit echt der Hammer.
andre_praises_herbs_2,Andre,"Diese frischen Kräuter, die da überall drin sind - einfach genial."
andre_praises_herbs_3,Andre,"Macht jedes Gericht zu etwas Besonderem, findest du nicht auch? Bitte mehr davon!"
fleur_complains_about_herbs_1,Fleur,"Mir ist aufgefallen, wie merkwürdig das Essen in der Taverne neuerdings schmeckt."
fleur_complains_about_herbs_2,Fleur,"Das liegt an Ritas Kräutern! Ich würde mich hüten, sowas einfach anzunehmen."
fleur_complains_about_herbs_3,Fleur,"Wer weiß schon, welche seltsamen Wirkungen die haben? Ich trau dem Braten nicht."
fleur_complains_about_herbs_4,Fleur,"Ich bitte dich, sei vorsichtig mit Hexengeschenken. Man kann nie wissen..."
maeve_affirmation_wealth_1,Maeve,"Ich weiß, dass die Zeiten hart sind und die Stadt mit finanziellen Sorgen zu kämpfen hat. Aber ich sehe, wie du dich anstrengst."
maeve_affirmation_wealth_2,Maeve,"Lass den Kopf nicht hängen, Schatz. Morgen ist ein neuer Tag, und ich glaube fest daran, dass alles besser wird. Wir stehen das zusammen durch."
maeve_affirmation_food_1,Maeve,"Ich hab gehört, dass unsere Vorräte schwinden, Liebling. Aber ich sehe, wie sehr du dich bemühst, das zu ändern."
maeve_affirmation_food_2,Maeve,"Denk dran, dass nach jedem Tief auch wieder ein Hoch kommt. Morgen sieht die Welt schon ganz anders aus. Ich glaube an dich."
maeve_affirmation_people_1,Maeve,"Ich sehe, wie unsere Besucher – und auch du – Sorgen mit sich tragen. Das Leben in der Stadt ist manchmal eben schwer."
maeve_affirmation_people_2,Maeve,"Aber denk dran, Herausforderungen sind da, um überwunden zu werden. Morgen kann alles ein bisschen heller aussehen. Ich bin für dich da."
maeve_affirmation_arms_1,Maeve,"Ich weiß, die Miliz gibt ihr Bestes, trotzdem sind da wohl Schwierigkeiten. Vielleicht ist es die Ausrüstung? Jeder gibt, was er kann, Schätzchen."
maeve_affirmation_arms_2,Maeve,"Aber denk immer daran, dass nach dem härtesten Sturm die Sonne wieder scheint. Ich bin mir sicher, morgen wird's ein bisschen besser. Wir halten zusammen, ja?"
maeve_customer_did_not_pay_1,Maeve,"Gestern war hier ein Gast in der Taverne, der ganz schön Ärger gemacht hat. Er hat getrunken, als gäbe es kein Morgen und dann einfach die Zeche geprellt!"
maeve_customer_did_not_pay_2,Maeve,"Es ist echt ärgerlich, wenn Leute nicht bezahlen."
maeve_customer_did_not_pay_tough,Chief,So ein Pech.
maeve_customer_did_not_pay_pay,Chief,Ich entschädige dich dafür.
maeve_militia_want_more_food_1,Maeve,"Die Soldaten haben mich neulich um größere Portionen gebeten, sie scheinen immer hungriger zu werden. Ich möchte sie natürlich nicht hungrig lassen."
maeve_militia_want_more_food_2,Maeve,"Darf ich vielleicht ein bisschen mehr von den Vorräten der Stadt verwenden, um sicherzustellen, dass alle satt werden?"
maeve_militia_want_more_food_yes,Chief,"Ja, gib ihnen mehr Essen."
maeve_militia_want_more_food_no,Chief,"Nein, sie bekommen bereits genug."
maeve_andre_complains_about_maeve_food_1,Andre,"Ich muss vorsichtig anmerken, dass einige der Männer bemerkt haben, dass die Portionen in Maeves Taverne sie nicht ganz satt machen."
maeve_andre_complains_about_maeve_food_2,Andre,"Es ist wichtig, dass jeder gestärkt in seinen Tag starten kann. Wir trainieren hart und benötigen entsprechend viel Energie."
bahri_donates_food_1,Bahri,"Nach einem geschäftigen Tag hatte ich viel unverkauftes Essen übrig. Früher war das immer ein Ärgernis, doch jetzt sehe ich es anders."
bahri_donates_food_2,Bahri,"Ich freue mich, dass ich die übrig gebliebenen Lebensmittel den bedürftigen Kindern spenden kann."
diegos_nightwatch_uneventful_1,Diego,"Nacht für Nacht streife ich durch die Straßen, leise und unsichtbar wie eine Katze auf der Jagd. Diese Stille, diese Dunkelheit – sie sind mein Reich, in dem ich mich auskenne."
diegos_nightwatch_uneventful_2,Diego,"Alles ist in Ordnung, keine Spur von Verbrechen oder Unrecht. Ich werde weiterhin wachsam sein..."
diego_found_coins_1,Diego,"Bei meiner letzten nächtlichen Patrouille habe ich etwas Unerwartetes gefunden. Versteckt in einer dunklen Ecke lag ein kleiner Sack, gefüllt mit Münzen."
diego_found_coins_2,Diego,"Ich frage mich jetzt, ob ich die Münzen behalten oder sie lieber der Stadt übergeben soll."
diego_found_coins_keep,Chief,"Wer es findet, darf es behalten."
diego_found_coins_town,Chief,Ich nehme die Münzen an mich.
diego_reports_sleeping_militia_1,Diego,"Du glaubst nicht, was ich letzte Nacht gesehen habe!"
diego_reports_sleeping_militia_2,Diego,"Auf meinem Abendspaziergang sah ich den Nachtwächter, im Tiefschlaf an den Stadttoren."
diego_reports_sleeping_militia_3,Diego,"Einfach unglaublich, in solchen Zeiten sollte jeder hellwach sein."
andre_reports_mysterious_person_1,Andre,"Gestern hat mir jemand erzählt, dass eine dunkle Gestalt nachts durch unsere Gassen schleicht. Das klingt beunruhigend, findest du nicht auch?"
andre_reports_mysterious_person_2,Andre,Die Menschen sind nervös wegen dieser unheimlichen Person.
andre_reports_mysterious_person_watch,Chief,Deine Männer sollen dort Wache halten.
andre_reports_mysterious_person_nonsense,Chief,So ein Quatsch.
fleur_offers_medicine_1,Fleur,"Diese Tropfen, hergestellt aus den weißen Blüten meines Gartens, wirken Wunder bei so gut wie allen Leiden."
fleur_offers_medicine_2,Fleur,"Ich biete sie der Stadt zu einem wirklich fairen Preis an. Na, wie klingt das?"
fleur_offers_medicine_yes,Chief,"Ja, wir kaufen die Tropfen."
fleur_offers_medicine_no,Chief,Nein danke.
rita_offers_medicine_1,Rita,"Ich habe eine Salbe aus Heilkräutern hergestellt, die effektiv viele Beschwerden mildert."
rita_offers_medicine_2,Rita,"Sie ist kein Wundermittel, aber ihre Linderung ist spürbar."
rita_offers_medicine_3,Rita,"Ich biete die Salbe zu einem fairen Preis der Stadt an, um Erleichterung zu bringen."
rita_offers_medicine_yes,Chief,"Ja, wir kaufen die Salbe."
rita_offers_medicine_no,Chief,Nein danke.
gunnar_isolates_sick_1,Gunnar,"Es melden sich nun immer mehr Kranke, {0}. Die Lage wird zunehmend kritischer. Ich rate dazu, die Erkrankten zur Sicherheit zu isolieren."
gunnar_isolates_sick_2,Gunnar,"Das mag den Menschen in der Stadt nicht gefallen, doch es wird die Ausbreitung eindämmen."
gunnar_isolates_sick_yes,Chief,"Ja, wir isolieren die Kranken."
gunnar_isolates_sick_no,Chief,"Nein, sie werden schon wieder gesund."
bahri_fever_spreads_1,Bahri,"Unter den Besuchern unseres Marktes gibt es immer mehr Fälle von Leuten, die unter trockenem Husten und entzündlich roten Augen leiden."
bahri_fever_spreads_2,Bahri,"Es scheint, als würde sich die Krankheit rapide ausbreiten."
bahri_fever_recedes_1,Bahri,"Ich habe bemerkt, dass endlich weniger Marktbesucher krank erscheinen."
bahri_fever_recedes_2,Bahri,"Das ist eine gute Nachricht, nicht nur für ihre Gesundheit, sondern auch für das Geschäft hier. Die Dinge blicken aufwärts!"
gunnar_refugees_sick_1,Gunnar,"Vor den Toren steht eine junge Frau, die schwach aussieht und gerötete Augen hat, {0}. Sie sucht bei uns Zuflucht."
gunnar_refugees_sick_2,Gunnar,"Es liegt in deiner Hand zu entscheiden, ob wir sie aufnehmen. Aber vergiss nicht, sie könnte ohne unsere Hilfe womöglich nicht überleben."
gunnar_refugees_sick_yes,Chief,"Ja, wir kümmern uns um sie."
gunnar_refugees_sick_no,Chief,"Nein, wir dürfen uns nicht bei ihr anstecken."
sigmund_buys_medicine_1,Sigmund,"Ich schlage vor, dass wir vielleicht Medizin aus der Stadt im Western besorgen sollten."
sigmund_buys_medicine_2,Sigmund,"Es scheint klug, alle Optionen zu prüfen, um die Ausbreitung weiter einzudämmen."
sigmund_buys_medicine_3,Sigmund,"Obwohl es wider mich geht, ist es vielleicht unsere beste Chance, dieser Plage Herr zu werden."
sigmund_buys_medicine_yes,Chief,"Ja, wir kaufen die Medizin."
sigmund_buys_medicine_no,Chief,"Nein, wir brauchen die Medizin nicht."
andre_fever_prevents_hunt_1,Andre,"Ich muss leider mitteilen, dass die geplante Jagd ausfallen muss. Zu viele meiner Männer leiden an der Krankheit."
andre_fever_prevents_hunt_2,Andre,"Ich entschuldige mich dafür, aber es wäre unverantwortlich und gefährlich, unter diesen Umständen auf die Jagd zu gehen."
clive_fever_prevents_harvest_1,Clive,"Ach {0}, es ist zum Verzweifeln. Das Gemüse auf den Feldern ist größtenteils verrottet."
clive_fever_prevents_harvest_2,Clive,"Die meisten meiner Erntehelfer liegen mit diesem schrecklichen Fieber im Bett. Ich weiß einfach nicht, wie es weitergehen soll."
maeve_offers_soup_1,Maeve,"Ich weiß, wie besorgt du wegen des Fiebers bist, {0}. Ich möchte auch beitragen, wo ich kann."
maeve_offers_soup_2,Maeve,"Vielleicht hilft es, wenn ich den Kranken eine stärkende Suppe bringe. Es ist keine Wunderkur, aber sie könnte zumindest etwas Linderung bringen."
maeve_offers_soup_3,Maeve,"Die Suppe kostet nicht viel, aber hoffentlich hilft sie den Leuten."
maeve_offers_soup_yes,Chief,Nimm etwas Geld für die Suppe.
maeve_offers_soup_no,Chief,Nein danke.
andre_demands_raise_for_militia_1,Andre,Meine Männer setzen sich Tag für Tag den Gefahren an der Stadtmauer aus. Sie kämpfen wirklich hart für unsere Sicherheit.
andre_demands_raise_for_militia_2,Andre,"Ehrlich, ich finde, sie verdienen mehr Anerkennung für ihre Mühen. Ein höherer Lohn wäre da nur gerecht."
andre_demands_raise_for_militia_3,Andre,"Ich würde mich freuen, wenn du darüber nachdenken könntest."
andre_demands_raise_for_militia_yes,Chief,"Ja, ich will sie gut bezahlen."
andre_demands_raise_for_militia_no,Chief,"Nein, sie verdienen genug."
sigmund_complains_about_raise_for_militia_1,Sigmund,"Ich finde es ziemlich besorgniserregend, dass du der Stadtwache zusätzliches Geld gibst."
sigmund_complains_about_raise_for_militia_2,Sigmund,"Sie schaffen es kaum, Recht und Ordnung hier zu erhalten. War das Geld wirklich nötig?"
sigmund_complains_about_raise_for_militia_3,Sigmund,"Vielleicht ist es an der Zeit, die wirklichen Prioritäten unserer Stadt zu überdenken, {0}."
pete_wants_money_for_hero_draught_1,Pete,Fleur hat mir ihren neuen Heldentrunk verkauft. Der gibt mir jedes Mal einen ordentlichen Schub!
pete_wants_money_for_hero_draught_2,Pete,"Nach jedem Schluck fühle ich mich wie neugeboren. Stärker und mutiger, als könnte ich Berge versetzen."
pete_wants_money_for_hero_draught_3,Pete,"Aber ich sag dir, das Zeug ist ganz schön teuer. Ein paar Münzen extra würden mir wirklich helfen, {0}."
pete_wants_money_for_hero_draught_yes,Chief,"Ja, aber trink nicht alles auf einmal."
pete_wants_money_for_hero_draught_no,Chief,"Nein, du brauchst diesen Trunk nicht."
andre_caught_pete_stealing_1,Andre,"Ich habe heute Pete beim Stehlen von Ausrüstung erwischt. Er versuchte offensichtlich, sie zu verkaufen."
andre_caught_pete_stealing_2,Andre,Ich habe ihm ordentlich die Meinung gesagt! Solches Verhalten können wir in unserer Stadt nicht tolerieren.
andre_caught_pete_stealing_3,Andre,"Ich hoffe wirklich, dass er daraus lernt. Es wäre schrecklich, wenn er so weitermacht. Bitte behalte das im Auge, {0}."
fleur_seeks_investor_1,Fleur,"Ich verkaufe Salben, Tinkturen und Tränke aller Art. Fast jeder hier in der Stadt liebt sie!"
fleur_seeks_investor_2,Fleur,"Ich hab da so eine Idee: Vielleicht möchtest du ja in mein kleines Geschäft investieren, {0}?"
fleur_seeks_investor_3,Fleur,"Du könntest am Gewinn teilhaben! Eine echte Win-win-Situation für uns beide, oder was meinst du?"
fleur_seeks_investor_yes,Chief,"Ja, ich investiere in dein Geschäft."
fleur_seeks_investor_no,Chief,"Nein, damit will ich nichts zu tun haben."
fleur_seeks_investor_double_down_1,Fleur,"Ich kann gar nicht genug danken, dass du an mein Geschäft geglaubt und investiert hast!"
fleur_seeks_investor_double_down_2,Fleur,"Ich spüre, dass der große Erfolg schon bald kommt, und dann wirst auch du belohnt werden."
fleur_seeks_investor_double_down_3,Fleur,"Doch um wirklich durchzustarten, benötige ich noch ein kleines bisschen mehr Kapital."
fleur_seeks_investor_double_down_4,Fleur,"Wie siehst du das, würdest du nochmal in meinen Traum investieren, {0}?"
fleur_seeks_investor_double_down_yes,Chief,"Okay, ich hoffe es zahlt sich aus!"
fleur_seeks_investor_double_down_no,Chief,Ich glaube das lohnt sich nicht...
fleur_seeks_investor_payout_1,Fleur,Ich habe fantastische Neuigkeiten! Meine Tränke und Salben sind der Renner!
fleur_seeks_investor_payout_2,Fleur,"Das bedeutet gute Gewinne, und wie versprochen, hier ist dein Anteil."
gunnar_complains_about_fleurs_business_1,Gunnar,"Es besorgt mich, wie gut Fleur ihre Tränke und Salben an die Leute bringt."
gunnar_complains_about_fleurs_business_2,Gunnar,"Ich glaube, ihr geht es weniger um das Wohl der Leute als um das Geld, das sie verdient."
gunnar_complains_about_fleurs_business_3,Gunnar,"Sie mag geschickt sein, aber man sollte stets auf die wahren Absichten achten."
clive_asks_for_vacation_1,Clive,"Die Arbeit auf den Feldern zehrt sehr an meinen Kräften, aber ich weiß, sie ist wichtig für die Stadt."
clive_asks_for_vacation_2,Clive,"Ohne meinen Einsatz wären viele hier ohne Essen, das ist mir klar."
clive_asks_for_vacation_3,Clive,"Ich wage es kaum zu fragen, aber könntest du mir ein paar Tage frei gewähren? Ich brauche dringend eine Pause."
clive_asks_for_vacation_yes,Chief,"Ja, erhole dich gut."
clive_asks_for_vacation_no,Chief,"Nein, vielleicht wann anders."
mary_ends_clives_vacation_1,Mary,"Es ist wirklich schön gewesen, Clive hier zu Hause zu haben. Die Kinder und ich haben jede Minute genossen."
mary_ends_clives_vacation_2,Mary,"Aber ich weiß auch, dass die anderen auf dem Feld auf seine Hilfe angewiesen sind."
mary_ends_clives_vacation_3,Mary,"Ich habe ihm deshalb gesagt, dass es Zeit ist, zurück an die Arbeit zu gehen. Das war nicht leicht, aber es musste sein."
rita_asks_for_protection_1,Rita,"Ich habe letzte Nacht zwei oder drei Männer bemerkt, die um mein Haus herumlungerten. Sie hatten Mistgabeln und Spaten dabei."
rita_asks_for_protection_2,Rita,"Natürlich machen mir solche Begegnungen Angst. Ich lebe allein hier draußen, und wer weiß schon, was sie vorhaben."
rita_asks_for_protection_3,Rita,"Kannst du einen der Milizsoldaten hierher schicken, um sie zu vertreiben? Das wäre mir wirklich eine große Hilfe, {0}."
rita_asks_for_protection_yes,Chief,"Ich schicke jemanden, der dich beschützt."
rita_asks_for_protection_no,Chief,Ich kann momentan niemanden entbehren.
bahri_caught_thief_1,Bahri,"Heute habe ich einen frechen Bengel erwischt, wie er einen Laib Brot von meinem Stand stahl!"
bahri_caught_thief_2,Bahri,"Wo ich herkomme, bestraft man jeden Dieb sofort und hart. Das hält die Ordnung aufrecht."
bahri_caught_thief_3,Bahri,"Wie handhaben wir das hier? Was soll ich jetzt tun, {0}?"
bahri_caught_thief_punish,Chief,Hau' ihm auf die Finger.
bahri_caught_thief_ignore,Chief,Lass den armen Jungen in Ruhe.
diego_spotted_logan_1,Diego,Auf meiner letzten Patrouille habe ich Logan und seine Truppe im Wald entdeckt. Sie haben dort ihr Lager aufgeschlagen.
diego_spotted_logan_2,Diego,"Kein Zweifel, sie bereiten sich darauf vor, bald hierher zu kommen und die Forderungen des Königs einzutreiben."
diego_spotted_logan_3,Diego,Wir sollten uns vorbereiten.
diego_spotted_logan_wealth,Chief,Wir sparen Geld.
diego_spotted_logan_food,Chief,Wir lagern Vorräte.
diego_spotted_logan_arms,Chief,Wir sammeln unsere Truppen.
maeve_donates_food_1,Maeve,"Ich habe in der Küche deutlich zu viel vorbereitet, mehr als meine Gäste essen können."
maeve_donates_food_2,Maeve,"Jetzt überlege ich, ob ich das überschüssige Essen lieber den tapferen Männern der Miliz überlassen oder den Armen der Stadt spenden sollte."
maeve_donates_food_3,Maeve,"Hilf mir bei der Entscheidung, {0}. Wem sollte ich das Essen geben?"
maeve_donates_food_militia,Chief,Gib es unserer Miliz.
maeve_donates_food_poor,Chief,Gib es den Armen.
rita_gifts_blueberries_1,Rita,"Schau mal, {0}. Ich habe einen Korb voll frischer Heidelbeeren gepflückt."
rita_gifts_blueberries_2,Rita,"Möchtest du sie für die Stadt haben? Ich denke, die Kinder würden sich darüber freuen."
rita_gifts_blueberries_yes,Chief,"Ja, die sehen lecker aus."
rita_gifts_blueberries_no,Chief,"Nein, lieber nicht."
maeve_made_blueberry_pie_1,Maeve,"Ich habe einen Heidelbeerkuchen gebacken, und er duftet einfach herrlich!"
maeve_made_blueberry_pie_2,Maeve,"Ich bin mir sicher, dass er den Leuten hier in der Stadt ausgezeichnet schmecken wird."
mary_complains_about_blueberries_1,Mary,"{0}, meine Kinder essen schon wieder Heidelbeeren und ich mache mir Sorgen."
mary_complains_about_blueberries_2,Mary,"Was ist, wenn wieder Tollkirschen dabei sind? Ich will nicht, dass sie sich vergiften."
andre_complains_about_blueberries_1,Andre,"Diese Kerle von mir, sie stopfen sich den Bauch mit Heidelbeeren voll, anstatt zu trainieren!"
andre_complains_about_blueberries_2,Andre,"Wie soll ich sie auf Vordermann bringen, wenn sie nur ans Naschen denken?"
pete_organizes_contest_1,Pete,Bald steigt wieder unser Wettkampf!
pete_organizes_contest_2,Pete,"Sollen wir die ganze Stadt begeistern, alles groß aufziehen – oder willst du’s diesmal ruhig halten, damit keiner zu wild wird?"
pete_organizes_contest_big,Chief,Großes Spektakel für alle!
pete_organizes_contest_small,Chief,Nur kleiner Wettkampfabend.
pete_organizes_contest_forbid,Chief,Ich sage den Wettkampf ab.
maeve_prepares_for_contest_1,Maeve,Zum nächsten Wettkampf will ich ein richtiges Festmahl anbieten.
maeve_prepares_for_contest_2,Maeve,"Reichen dir Brot und Eintopf, oder sollen die Gäste mal richtig schlemmen?"
maeve_prepares_for_contest_big,Chief,Mach ein Festmahl!
maeve_prepares_for_contest_small,Chief,Bleib bei einfacher Kost.
clive_won_contest_1,Clive,"Ich hab beim Wettkampf gewonnen. Schwer war’s schon, aber am Ende zählt, wer durchhält."
gunnar_won_contest_1,Gunnar,"Der Wettkampf gestern hat mir gezeigt, dass ich noch nicht zum alten Eisen gehöre."
gunnar_won_contest_2,Gunnar,"Es tut gut, die jungen Hüpfer mal wieder in die Schranken zu weisen."
logan_won_contest_1,Logan,"Der Wettkampf? Ich war gar nicht eingeladen, bin trotzdem rein und hab alle abgezogen."
logan_won_contest_2,Logan,"Deine ""Männer"" sollen ruhig wissen, dass mit mir nicht zu spaßen ist."
pete_lost_contest_1,Pete,Der Wettkampf war hart – und ich wurde wieder übertroffen. Aber es war sooo knapp!
pete_lost_contest_2,Pete,"Irgendwann hol ich mir auch mal den Sieg, das schwör ich!"
andres_men_after_contest_1,Andre,Nach dem letzten Wettkampf sind einige Männer noch nicht wieder auf den Beinen.
andres_men_after_contest_2,Andre,Sollen sie sich ausruhen oder gleich wieder zum Dienst antreten?
andres_men_after_contest_rest,Chief,Lass sie ausruhen.
andres_men_after_contest_work,Chief,Alle zurück zur Arbeit.
clive_builds_fence_1,Clive,"{0}, die Ziegen haben schon wieder das frische Gemüse vom Beet gefressen."
clive_builds_fence_2,Clive,"Ich muss den Zaun reparieren, aber das kostet unser bestes Holz…"
clive_builds_fence_3,Clive,"Wenn wir sie laufen lassen, riskieren wir weniger Ernte."
clive_builds_fence_4,Clive,Was meinst du?
clive_builds_fence_agree,Chief,Baue einen Zaun.
clive_builds_fence_disagree,Chief,Dafür haben wir nicht genug Holz.
andre_pays_the_blacksmith_1,Andre,"{0}, unsere Rüstungen und Schwerter sind nicht mehr das, was sie mal waren."
andre_pays_the_blacksmith_2,Andre,"Der Schmied kann sie reparieren aber wie du weißt, lässt er sich gut dafür bezahlen."
andre_pays_the_blacksmith_3,Andre,Bezahlst du den Schmied oder sollen wir improvisieren?
andre_pays_the_blacksmith_pay,Chief,Ich bezahle den Schmied.
andre_pays_the_blacksmith_reject,Chief,Eure Ausrüstung ist noch gut genug.
bahri_gambles_1,Bahri,"Ein paar reisende Händler haben mich zu einem Kartenspiel eingeladen. Der Pot wär groß, aber das Risiko hoch."
bahri_gambles_2,Bahri,Zahlst du mir den Einsatz? Mit etwas Glück können wir unsere Schatzkammer füllen!
bahri_gambles_yes,Chief,"Ja, hoffentlich gewinnst du."
bahri_gambles_no,Chief,"Nein, das ist zu riskant."
bahri_gambles_win_1,Bahri,Was für ein Abend! Die Karten haben zu mir gesprochen – ich habe den Pot geknackt!
bahri_gambles_win_2,Bahri,"Unsere Kasse ist praller als je zuvor. Ein bisschen Mut, ein kleiner Einsatz, und schon floriert unser Wohlstand."
bahri_gambles_lose_1,Bahri,"Tja, diesmal hatte ich wohl nicht die besten Karten. Das Gold ist futsch, tut mir Leid."
bahri_gambles_lose_2,Bahri,Aber ich geb’s zu: Die Spannung war es fast wert! Beim nächsten Mal gewinne ich ganz bestimmt.
bahri_gambles_debt_1,Bahri,"{0}... äh... also, ich wollte ja nur das Beste für unser Dorf, aber das Glück war mir gar nicht hold."
bahri_gambles_debt_2,Bahri,Nicht nur das eingesetzte Gold ist weg – ich hab’ obendrein Schulden bei den Händlern gemacht.
bahri_gambles_sigmund_complains_1,Sigmund,Bahri ist ein Narr! Mit dem hart erarbeiteten Gold der Stadt spielt man nicht leichtfertig.
bahri_gambles_sigmund_complains_2,Sigmund,Glücksspiel ist der erste Schritt ins Verderben – nichts als Gier und Trug.
bahri_gambles_sigmund_complains_3,Sigmund,"Wer Wohlstand erhalten will, vertraut auf Fleiß, nicht auf das blinde Ziehen einer Karte."
bahri_gambles_sigmund_joins_1,Sigmund,"Du weißt, ich verachte das Glücksspiel. Aber wenn du es Bahri erlaubst, dann werde ich auch mitspielen!"
bahri_gambles_sigmund_joins_2,Sigmund,Vielleicht zieht das Glück ja sogar einen Altmeister wie mich einmal aus dem Schatten.
bahri_gambles_sigmund_lost_1,Sigmund,"Ich hab’s gewusst – Glücksspiel ist nichts als ein raffinierter Weg, ehrliche Leute um ihr Geld zu bringen."
bahri_gambles_sigmund_lost_2,Sigmund,"Kein Wunder, dass ich verloren habe! Wer auf Glück vertraut, ist ein Narr."
bahri_gambles_sigmund_lost_3,Sigmund,"Beim nächsten Mal halte ich mich an das, was ich immer gesagt habe: Finger weg davon!"
fleur_buys_candles_1,Fleur,"{0}, ich muss darauf bestehen: In meinem Arbeitszimmer herrscht schon wieder Halbdunkel!"
fleur_buys_candles_2,Fleur,Wie soll ein Geist wie meiner unter solchen Bedingungen genial sein?
fleur_buys_candles_3,Fleur,"Ich brauche noch zwei, drei Kerzen mehr – das ist doch wirklich nicht zu viel verlangt."
fleur_buys_candles_yes,Chief,Kaufe dir ein paar Kerzen.
fleur_buys_candles_no,Chief,Arbeite doch bei Tageslicht.

RESSOURCEN (immer im Blick behalten):
- Vermögen (Geld)
- Zufriedenheit (Bevölkerung)
- Nahrung (Essen & Trinken)
- Waffen (Rüstung)

SPIELMECHANIK:
- Wähle 1 zufälligen Charakter
- Der Charakter erklärt dem Spieler ein Bedürfnis oder ein Problem
- Biete 2 Entscheidungsoptionen aus Sicht des Spielers (wörtliche Rede)
- Variiere die ausgewählten Charaktere und erfinde ständig neue lösbare Probleme. Sie sollten sich nicht wiederhohlen.
- Wiederhole keine Situationen aus den TEXTBEISPIELEN oben.
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