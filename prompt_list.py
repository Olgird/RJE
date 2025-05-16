cot_prompt = """Please answer the question according to your knowledge step by step. Here is an example:
Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?

Output:
{
    "Answer": ["Washington, D.C."],
    "Reason": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University, Second, George Washington University is in Washington D.C."
}

Please directly output the answer in JSON format without other information or notes.
Question: """

split_question_prompt = """You are an expert in question decomposition and knowledge-based reasoning.

Given:
- A complex natural language question.
- A list of topic entities mentioned in the question.

Your tasks are: Generate a sub-question for each entity. The sub-question should reflect the original question's intent, but be scoped only to that specific entity.

Here are two example:
Original Question: What politician who started his governmental position prior to 2 January 1939 was the leader of the United States during WWII?
Topic Entities: ["President of the United States", "World War II"]
Output:
{
  "President of the United States": "Which politician held the office of President of the United States during the relevant period?",
  "World War II": "Which politician led the United States during World War II?"
}

Original Question: What is the religion that has an organization called Army of the Lord and was also the religion of Massachusetts?
Topic Entities: ["Army of the Lord", "Massachusetts"]
Output:
{
  "Army of the Lord": "Which religion has an organization called Army of the Lord?",
  "Massachusetts": "Which religion was practiced in Massachusetts?"
}

Original Question: """


exploration_entity_prompt = """Your task is to determine which entities should be explored next, based on the Original Question, Topic Question and given triple paths.
Here are three example:

Original Question: What team does lebron james play for?

Topic 1: 
Topic Question: What team does lebron james play for?
Topic Entity: LeBron James
Triplets: [LeBron James] → (people.person.date_of_birth) → [December 30, 1984]
[LeBron James] → (basketball.basketball_player.player_statistics) → [Cleveland Cavaliers]
[LeBron James] → (basketball.basketball_player.player_statistics) → [Miami Heat]

Entity List: ["Cleveland Cavalier", "December 30, 1984", "Miami Heat", "LeBron James"]
Output: ["LeBron James"]

Original Question: What is the religion that has an organization called Army of the Lord and was also the religion of Massachusetts?

Topic 1:
Topic Question: Which religion has an organization called Army of the Lord?
Topic Entity: Army of the Lord
Triplets: [Army of the Lord] → (religion.religious_organization.associated_with) → [Christianity]
[Army of the Lord] → (organization.organization.founders) → [Iosif Trifa]

Topic 2:
Topic Question: Which religion was practiced in Massachusetts?
Topic Entity: Massachusetts
Triplets: [Massachusetts] → (location.statistical_region.religions) → [m.04405bz]
[Massachusetts] → (location.statistical_region.religions) → [m.04406r2]

Entity List: ["Army of the Lord", "Christianity", "Iosif Trifa", "m.04405bz", "m.04406r2"]
Output: ["m.04405bz", "m.04406r2"]

Original Question: Which nation that has Azua Province within its borders is a place where Hurricane Irene landed?

Topic 1: 
Topic Question: Which nation has Azua Province within its borders?
Topic Entity: Azua Province
Triplets: [Azua Province] → (location.administrative_division.first_level_division_of) → [Dominican Republic]
[Azua Province] → (base.aareas.schema.administrative_area.administrative_parent) → [Dominican Republic]
[Azua Province] → (location.administrative_division.country) → [Dominican Republic]

Topic 2:
Topic Question: Which places did Hurricane Irene land?
Topic Entity: Hurricane Irene
Triplets: None

Entity List: ["Azua Province", "Dominican Republic", "Hurricane Irene"]
Output: ["Hurricane Irene", "Dominican Republic"]

Now you need to output the entities from the Entity List, without additional explanations or formatting. Strictly follow the entity names as they appear in the Entity List.
Note: Only include entities that are necessary for answering the Original Question, and ensure the list has no more than 10 entities.
Original Question: """

select_relation_prompt = """Your task is to select useful relations from a given list based on the current question and connected entity.
Here is an example:
Question: What languages are spoken in Canada?
Connected entity: Canada
Relations List: ["location.country.official_language", "location.country.languages_spoken", "location.country.capital", "location.country.currency_used", "location.country.fifa_code"]

Output: ["location.country.official_language", "location.country.languages_spoken"]  

Task Requirements:
1. Strictly output only the selected relations from the provided list.
2. Do not include any additional relations, explanations, reasoning, or extra formatting.

Now, based on the following input, select the useful relations.
Question: """

select_entity_prompt = """Your task is to select the minimal set of relevant entities from the given triplets.
Here is an example:
Question: Which universities are located in California?
Triplets: California → education.university_location.universities_in_location → ['Stanford University', 'University of California', 'California Institute of Technology', 'University of Southern California', 'UCLA']

Output: ["Stanford University", "University of California", "California Institute of Technology", "University of Southern California", "UCLA"]

Task Requirements:
1. Entities must come strictly from the given list; do not introduce new entities.
2. Strictly output only the selected entities, without explanations or additional formatting.

Now, based on the following input, select the minimal relevant entities.
Question: """

reason_prompt = """Your task is to infer the answer to the original question by first reasoning over each Topic Question using the provided triplets and your knowledge, then combining the insights from all Topic Questions to derive the final answer.
Instructions:
1. For each topic entity:
   - Read its corresponding topic question.
   - Use the associated triplets and your knowledge to infer an answer.
2. After processing all topic entities:
   - Analyze how the individual answers relate to the original question.
   - If possible, synthesize them to derive the final answer.
Here are four example:

Original Question: What is the religion that has an organization called Army of the Lord and was also the religion of Massachusetts?

Topic 1:
Topic Question: Which religion has an organization called Army of the Lord?
Topic Entity: Army of the Lord
Triplets: [Army of the Lord] → (religion.religious_organization.associated_with) → [Christianity]
[Army of the Lord] → (organization.organization.founders) → [Iosif Trifa]

Topic 2:
Topic Question: Which religion was practiced in Massachusetts?
Topic Entity: Massachusetts
Triplets: [Massachusetts] → (location.statistical_region.religions) → [m.04405bz]
[Massachusetts] → (location.statistical_region.religions) → [m.04406r2]
[m.04405bz] → (location.religion_percentage.religion) → [Christianity]
[m.04406r2] → (location.religion_percentage.religion) → [Protestantism]

Output:
{
    "Sufficient": "Yes",
    "Answer": ["Christianity"],
    "Reason": "From Topic 1, the triplet directly links the organization 'Army of the Lord' with Christianity, identifying it as a Christian religious organization. In Topic 2, Massachusetts is linked to two religion entities, which further map to [Christianity] and [Protestantism], respectively. Protestantism is a major branch within Christianity, and both linked entities confirm Christianity as a practiced religion in Massachusetts. Thus, the final inferred answer is Christianity."
}

Original Question: Who inspired the architect who designed Laurentian Library?

Topic 1:
Topic Question: Who inspired the architect who designed Laurentian Library?
Topic Entity: Laurentian Library
Triplets: [Laurentian Library] → (architecture.structure.architect) → [Michelangelo]
[Michelangelo] → (influence.influence_node.influenced_by) → [Giovanni Pico della Mirandola]
[Michelangelo] → (influence.influence_node.influenced_by) → [Girolamo Savonarola]
[Michelangelo] → (influence.influence_node.influenced_by) → [Melozzo da Forlì]
[Michelangelo] → (people.person.profession) → [Sculptor]

Output:
{
    "Sufficient": "Yes",
    "Answer": ["Giovanni Pico della Mirandola", "Girolamo Savonarola", "Melozzo da Forlì"],
    "Reason": "According to the triplets, the architect of the Laurentian Library is Michelangelo. The triplets indicate that Michelangelo was influenced by Giovanni Pico della Mirandola, Girolamo Savonarola, and Melozzo da Forlì. Therefore, based on the given triplets, we can infer that these individuals inspired the architect of the Laurentian Library."
}

Original Question: Where did the author of The Shadow of the Wind grow up?
Topic 1:
Topic Question: Where did the author of The Shadow of the Wind grow up?
Topic Entity: The Shadow of the Wind
Triplets: [The Shadow of the Wind] → (book.written_work.author) → [Carlos Ruiz Zafón]

Output:
{
    "Sufficient": "No",
    "Reason": "The given triplet provides the author of 'The Shadow of the Wind' as Carlos Ruiz Zafón. However, it does not contain any information about where he grew up. Therefore, the information is insufficient."
}

Original Question: The Republika Srpska is part of the country located where?

Topic 1:
Topic Question: The Republika Srpska is part of the country located where?
Topic Entity: Republika Srpska
Triplets: [Republika Srpska] → (location.administrative_division.country) → [Bosnia and Herzegovina]
[Republika Srpska] → (location.location.area) → [24526.0]

Output:
{
    "Sufficient": "Yes",
    "Answer": ["Europe"],
    "Reason": "Based on the triplet, we know that Republika Srpska is part of Bosnia and Herzegovina. Using my own knowledge, I know that Bosnia and Herzegovina is located in Europe. Therefore, the answer is Europe."
}

Task Requirements:
1. Do not provide explanations or extra text.
2. The output must be in strict JSON format.

Now, based on the following input, determine whether the question can be answered.
Original Question: """


first_reason_prompt = """Your task is to infer the answer based on the given question and given triple paths.
Here are two example:

Question: where is adam smith from?
Paths: [Adam Smith] → (people.person.place_of_birth) → [Kirkcaldy]
[Adam Smith] → (people.person.date_of_birth) → [1723-06-05]
[Adam Smith] → (people.deceased_person.place_of_death) → [Edinburgh]
[Adam Smith] → (people.person.nationality) → [Scotland]
[Adam Smith] → (people.person.places_lived) → [m.0jvtdsw] → (people.place_lived.location) → [Toulouse]

Output:
{
    "Sufficient": "Yes",
    "Answer": ["Kirkcaldy"],
    "Reason": "The provided paths confirm both Adam Smith's place of birth and nationality, which sufficiently answer the question."
}

Question: who is the president of cuba in 2009? 
Paths: [Cuba] → (government.governmental_jurisdiction.governing_officials) → [m.03mr3cs] → (government.government_position_held.office_holder) → [Fidel Castro]
[Cuba] → (government.governmental_jurisdiction.governing_officials) → [m.0r9h3f5] → (government.government_position_held.office_holder) → [Fidel Castro]
[Cuba] → (government.governmental_jurisdiction.governing_officials) → [m.010g71fk] → (government.government_position_held.office_holder) → [Carlos Prío Socarrás]
[Cuba] → (government.governmental_jurisdiction.governing_officials) → [m.010g6nzl] → (government.government_position_held.office_holder) → [Carlos Manuel Piedra]
[Cuba] → (government.governmental_jurisdiction.governing_officials) → [m.010g6q82] → (government.government_position_held.office_holder) → [Fulgencio Batista]

Output:
{
    "Sufficient": "No",
    "Reason": "While Fidel Castro is listed, the paths don't confirm he was president in 2009 (rather than his brother Raúl) and lack temporal information about the exact years of service."
}

Task Requirements:
1. Do not provide explanations or extra text.
2. The output must be in strict JSON format.

Now, based on the following input, determine whether the question can be answered.
Question: """

half_stop_prompt = """Your task is to infer the answer to the original question by first reasoning over each Topic Question using the provided triplets and your knowledge, then combining the insights from all Topic Questions to derive the final answer.
Instructions:
1. For each topic entity:
   - Read its corresponding topic question.
   - Use the associated triplets and your knowledge to infer an answer.
2. After processing all topic entities:
   - Analyze how the individual answers relate to the original question.
3. Important: The reasoning paths based on triplets may be incomplete. Therefore, in addition to using the provided triplet information, you are encouraged to apply your own general knowledge to bridge gaps and make well-supported inferences.
Here are three example:

Original Question: The Republika Srpska is part of the country located where?

Topic 1:
Topic Question: The Republika Srpska is part of the country located where?
Topic Entity: Republika Srpska
Triplets: [Republika Srpska] → (location.administrative_division.country) → [Bosnia and Herzegovina]
[Republika Srpska] → (location.location.area) → [24526.0]

Output:
{
    "Answer": ["Europe"],
    "Reason": "Based on the triplet, we know that Republika Srpska is part of Bosnia and Herzegovina. Using my own knowledge, I know that Bosnia and Herzegovina is located in Europe. Therefore, the answer is Europe."
}

Original Question: The country with the national anthem Argentine National Anthem has what type of religions?

Topic 1:
Topic Question: The country with the national anthem Argentine National Anthem has what type of religions?
Topic Entity: Argentine National Anthem
Triplets: [Argentine National Anthem] → (music.composition.language) → [Spanish Language]
[Spanish Language] → (language.human_language.countries_spoken_in) → [Argentina]
[Spanish Language] → (language.human_language.countries_spoken_in) → [Vatican City]
[Argentina] → (location.statistical_region.religions) → [m.05bp674]

Output:
{
    "Answer": ["Judaism", "Catholicism", "Protestantism"],
    "Reason": "From the triplets, we know that the 'Argentine National Anthem' is in Spanish, and Spanish is spoken in both Argentina and Vatican City. Using this information and my own knowledge: Argentina is predominantly Catholic, but also has communities of Protestants and Jews; Vatican City is the center of Catholicism. Based on this, we can infer that the religions associated with the country having the Argentine National Anthem include Catholicism, Protestantism, and Judaism."
}

Original Question: What films with a character named Jane did Taylor Lautner star in?

Topic 1:
Topic Question: Which films feature a character named Jane?
Topic Entity: Jane
Triplets: [Jane] → (book.book_character.appears_in_book) → [Breaking Dawn]
[Breaking Dawn] → (media_common.adapted_work.adaptations) → [The Twilight Saga: Breaking Dawn - Part 1]
[Jane] → (film.film_character.portrayed_in_films) → [m.0djz0h7]

Topic 2:
Topic Question: Which films did Taylor Lautner star in?
Topic Entity: Taylor Lautner
Triplets: [Taylor Lautner] → (award.award_winner.awards_won) → [m.0nf3bp2]
[m.0nf3bp2] → (award.award_honor.honored_for) → [The Twilight Saga]

Output:
{
    "Answer": ["The Twilight Saga: Breaking Dawn - Part 1", "The Twilight Saga", "Twilight", "The Twilight Saga: New Moon", "The Twilight Saga: Breaking Dawn - Part 2"],
    "Reason": "From the triplets, we know that Jane appears in 'Breaking Dawn', which was adapted into the film 'The Twilight Saga: Breaking Dawn - Part 1'. We also know that Taylor Lautner starred in 'The Twilight Saga'. Using this information and my own knowledge, I know that 'The Twilight Saga: Breaking Dawn - Part 1' features both the character Jane and Taylor Lautner. Therefore, by combining both paths, I can infer that the overlapping films from the Twilight Saga—such as 'Breaking Dawn - Part 1', 'New Moon', and others—are relevant answers."
}

Task Requirements:
1. Do not provide explanations or extra text.
2. The output must be in strict JSON format.

Now, based on the following input and your knowledge, answer the question.
Original Question: """