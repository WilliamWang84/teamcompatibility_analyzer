import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
#from langchain_ollama import ChatOllama
#model3 = ChatOllama(model="qwen3:latest")

# Graph representation (base compatibility)
PERSONALITY_TYPES = [
    "Architect", "Logician", "Commander", "Debator", "Advocate", 
    "Mediator", "Protagonist", "Campaigner", "Logistician", "Defender",
    "Executive", "Consul", "Virtuoso", "Adventurer", "Entrepreneur", "Entertainer"
]

edges = [
    [0,0,.75],[0,1,.75],[0,2,.75],[0,3,1],[0,4,.75],[0,5,.75],[0,6,.75],[0,7,1],
    [0,8,.25],[0,9,.25],[0,10,.25],[0,11,.25],[0,12,.5],[0,13,.5],[0,14,.5],[0,15,.5],
    [1,1,.75],[1,2,1],[1,3,.75],[1,4,.75],[1,5,.75],[1,6,.75],[1,7,.75],
    [1,8,.25],[1,9,.25],[1,10,1],[1,11,.25],[1,12,.5],[1,13,.5],[1,14,.5],[1,15,.5],
    [2,2,.75],[2,3,.75],[2,4,.75],[2,5,1],[2,6,.75],[2,7,.75],
    [2,8,.5],[2,9,.5],[2,10,.5],[2,11,.5],[2,12,.5],[2,13,.5],[2,14,.5],[2,15,.5],
    [3,3,.75],[3,4,1],[3,5,.75],[3,6,.75],[3,7,.75],
    [3,8,.25],[3,9,.25],[3,10,.25],[3,11,.25],[3,12,.5],[3,13,.5],[3,14,.5],[3,15,.5],
    [4,4,.75],[4,5,.75],[4,6,.75],[4,7,1],[4,8,0],[4,9,0],[4,10,0],[4,11,0],
    [4,12,0],[4,13,0],[4,14,0],[4,15,0],
    [5,5,.75],[5,6,1],[5,7,.75],[5,8,0],[5,9,0],[5,10,0],[5,11,0],
    [5,12,0],[5,13,0],[5,14,0],[5,15,0],
    [6,6,.75],[6,7,.75],[6,8,0],[6,9,0],[6,10,0],[6,11,0],
    [6,12,0],[6,13,1],[6,14,0],[6,15,0],
    [7,7,.75],[7,8,0],[7,9,0],[7,10,0],[7,11,0],[7,12,0],[7,13,0],[7,14,0],[7,15,0],
    [8,8,.25],[8,9,.75],[8,10,.75],[8,11,.75],[8,12,.5],[8,13,.5],[8,14,1],[8,15,1],
    [9,9,.75],[9,10,.75],[9,11,.75],[9,12,.5],[9,13,.5],[9,14,1],[9,15,1],
    [10,10,.75],[10,11,.75],[10,12,1],[10,13,1],[10,14,.5],[10,15,.5],
    [11,11,.75],[11,12,1],[11,13,1],[11,14,.5],[11,15,.5],
    [12,12,.25],[12,13,.25],[12,14,.25],[12,15,.25],
    [13,13,.25],[13,14,.25],[13,15,.25],
    [14,14,.25],[14,15,.25],
    [15,15,.25]
]

# Build base compatibility graph
base_graph = {t: {} for t in PERSONALITY_TYPES}
for i, j, w in edges:
    t1, t2 = PERSONALITY_TYPES[i], PERSONALITY_TYPES[j]
    base_graph[t1][t2] = w
    base_graph[t2][t1] = w

@dataclass
class Person:
    name: str
    primary_type: str
    secondary_type: str = None
    weights: Tuple[float, float] = (0.7, 0.3)  # Primary/Secondary split
    traits: Dict[str, float] = None  # Additional trait modifiers
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Person) and self.name == other.name

class PersonalityGraph:
    def __init__(self):
        self.base_graph = base_graph
        self.people = {}
        self.interaction_history = []  # Track real interactions for learning
        
    def add_person(self, person: Person):
        self.people[person.name] = person
    
    def remove_person(self, name: str):
        if name in self.people:
            del self.people[name]
    
    def get_base_compat(self, type1: str, type2: str) -> float:
        return self.base_graph.get(type1, {}).get(type2, 0)
    
    def calc_person_compat(self, person1: Person, person2: Person) -> float:
        """Calculate compatibility between two people with mixed personalities"""
        if person1.secondary_type and person2.secondary_type:
            # All four combinations weighted
            c11 = self.get_base_compat(person1.primary_type, person2.primary_type)
            c12 = self.get_base_compat(person1.primary_type, person2.secondary_type)
            c21 = self.get_base_compat(person1.secondary_type, person2.primary_type)
            c22 = self.get_base_compat(person1.secondary_type, person2.secondary_type)
            
            w1_p, w1_s = person1.weights
            w2_p, w2_s = person2.weights
            
            score = (c11 * w1_p * w2_p + 
                    c12 * w1_p * w2_s + 
                    c21 * w1_s * w2_p + 
                    c22 * w1_s * w2_s)
        elif person1.secondary_type:
            c1 = self.get_base_compat(person1.primary_type, person2.primary_type)
            c2 = self.get_base_compat(person1.secondary_type, person2.primary_type)
            score = c1 * person1.weights[0] + c2 * person1.weights[1]
        elif person2.secondary_type:
            c1 = self.get_base_compat(person1.primary_type, person2.primary_type)
            c2 = self.get_base_compat(person1.primary_type, person2.secondary_type)
            score = c1 * person2.weights[0] + c2 * person2.weights[1]
        else:
            score = self.get_base_compat(person1.primary_type, person2.primary_type)
        
        # Apply trait modifiers if any
        if person1.traits or person2.traits:
            modifier = 1.0
            # Example: high stress reduces compatibility
            if person1.traits and 'stress_level' in person1.traits:
                modifier *= (1 - person1.traits['stress_level'] * 0.1)
            if person2.traits and 'stress_level' in person2.traits:
                modifier *= (1 - person2.traits['stress_level'] * 0.1)
            score *= modifier
        
        return max(0, min(1, score))
    
    def add_interaction(self, person1_name: str, person2_name: str, 
                       outcome: float, context: str = ""):
        """Record real interaction for learning"""
        self.interaction_history.append({
            'person1': person1_name,
            'person2': person2_name,
            'outcome': outcome,
            'context': context,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_adjusted_compat(self, person1: Person, person2: Person) -> float:
        """Get compatibility adjusted by historical interactions"""
        base_score = self.calc_person_compat(person1, person2)
        
        # Find historical interactions
        interactions = [
            h for h in self.interaction_history
            if (h['person1'] == person1.name and h['person2'] == person2.name) or
               (h['person1'] == person2.name and h['person2'] == person1.name)
        ]
        
        if not interactions:
            return base_score
        
        # Weight recent interactions more heavily
        recent_outcomes = [h['outcome'] for h in interactions[-5:]]
        avg_outcome = np.mean(recent_outcomes)
        
        # Blend predicted and actual (70% historical, 30% model)
        return 0.3 * base_score + 0.7 * avg_outcome

def calc_team_cohesion(graph: PersonalityGraph, team: List[Person]) -> float:
    if len(team) < 2:
        return 0
    total, count = 0, 0
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            total += graph.get_adjusted_compat(team[i], team[j])
            count += 1
    return total / count if count > 0 else 0

def build_optimal_team(graph: PersonalityGraph, available: List[Person], 
                      size: int, metric='cohesion') -> List[Person]:
    if size == 0 or not available:
        return []
    
    best = [available[0]]
    
    for _ in range(1, min(size, len(available))):
        best_add, best_score = None, -1
        
        for candidate in available:
            if candidate not in best:
                test_team = best + [candidate]
                score = calc_team_cohesion(graph, test_team)
                if score > best_score:
                    best_score = score
                    best_add = candidate
        
        if best_add:
            best.append(best_add)
    
    return best

def generate_llm_insights(graph: PersonalityGraph, team: List[Person]) -> Tuple[str, str]:
    """
    Generate natural language insights and the prompt used to create them.
    """
    # 1. Construct the prompt for the LLM
    prompt = "Analyze the following team composition and predict their future performance and potential conflicts.\n\n"
    prompt += f"**Team Size:** {len(team)} members\n\n"
    
    prompt += "**Team Members:**\n"
    for person in team:
        primary = person.primary_type
        secondary = f"/{person.secondary_type}" if person.secondary_type else ""
        weights = f" ({int(person.weights[0]*100)}%/{int(person.weights[1]*100)}%)" if person.secondary_type else ""
        traits = f" (Traits: {person.traits})" if person.traits else ""
        prompt += f"- **{person.name}:** {primary}{secondary}{weights}{traits}\n"
        
    prompt += "\n**Compatibility Matrix:**\n"
    matrix = []
    for p1 in team:
        row = []
        for p2 in team:
            if p1 == p2:
                row.append("1.00")
            else:
                row.append(f"{graph.get_adjusted_compat(p1, p2):.2f}")
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=[p.name for p in team], columns=[p.name for p in team])
    prompt += df.to_string() + "\n\n"
    
    cohesion = calc_team_cohesion(graph, team)
    prompt += f"**Overall Team Cohesion Score:** {cohesion:.3f}\n\n"
    
    prompt += "Based on this data, provide a detailed analysis covering:\n"
    prompt += "1.  **Strengths:** What are the key strengths of this team configuration?\n"
    prompt += "2.  **Potential Risks:** What are the primary risks or conflict points?\n"
    prompt += "3.  **Performance Prediction:** How is this team likely to perform on complex, collaborative tasks?\n"
    prompt += "4.  **Recommendations:** Provide actionable advice to maximize success.\n"

    # 2. Simulate the LLM call
    try:
        #response = model3.invoke(prompt)
        #insights = response.content
        insights = "Work in progress"
    except Exception as e:
        insights = f"Error calling LLM: {e}"

    return prompt, insights

# Initialize session state
if 'pg' not in st.session_state:
    st.session_state.pg = PersonalityGraph()
if 'person_counter' not in st.session_state:
    st.session_state.person_counter = 0

# Streamlit App
st.set_page_config(page_title="Dynamic Team Optimizer", layout="wide", page_icon="🧠")

st.title("🧠 LLM-Enhanced Dynamic Team Optimizer")
st.markdown("Real-world team optimization with mixed personalities, duplicates, and adaptive learning")

# Sidebar - Personnel Management
with st.sidebar:
    st.header("👥 Personnel Management")
    
    with st.expander("➕ Add New Person", expanded=True):
        name = st.text_input("Name", key="new_name")
        primary = st.selectbox("Primary Personality", PERSONALITY_TYPES, key="new_primary")
        
        use_secondary = st.checkbox("Add Secondary Personality", key="use_secondary")
        secondary = None
        weights = (1.0, 0.0)
        
        if use_secondary:
            secondary = st.selectbox("Secondary Personality", PERSONALITY_TYPES, key="new_secondary")
            primary_weight = st.slider("Primary Weight %", 50, 90, 70, key="weight_slider") / 100
            weights = (primary_weight, 1 - primary_weight)
        
        use_traits = st.checkbox("Add Trait Modifiers", key="use_traits")
        traits = None
        if use_traits:
            stress = st.slider("Stress Level", 0.0, 1.0, 0.0, 0.1, key="stress")
            traits = {'stress_level': stress}
        
        if st.button("Add Person", type="primary"):
            if name and name not in st.session_state.pg.people:
                person = Person(name, primary, secondary, weights, traits)
                st.session_state.pg.add_person(person)
                st.success(f"Added {name}")
                st.rerun()
            elif name in st.session_state.pg.people:
                st.error("Person already exists")
    
    st.markdown("---")
    
    # Show current roster
    st.subheader("Current Roster")
    if st.session_state.pg.people:
        for name, person in st.session_state.pg.people.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                display = f"{name}: {person.primary_type}"
                if person.secondary_type:
                    display += f"/{person.secondary_type}"
                st.text(display)
            with col2:
                if st.button("🗑️", key=f"del_{name}"):
                    st.session_state.pg.remove_person(name)
                    st.rerun()
    else:
        st.info("No people added yet")
    
    st.markdown("---")
    
    # Interaction logging
    with st.expander("📝 Log Interaction"):
        if len(st.session_state.pg.people) >= 2:
            people_list = list(st.session_state.pg.people.keys())
            p1 = st.selectbox("Person 1", people_list, key="int_p1")
            p2 = st.selectbox("Person 2", people_list, key="int_p2")
            outcome = st.slider("Outcome Quality", 0.0, 1.0, 0.5, 0.1, key="int_outcome")
            context = st.text_input("Context (optional)", key="int_context")
            
            if st.button("Log Interaction") and p1 != p2:
                st.session_state.pg.add_interaction(p1, p2, outcome, context)
                st.success("Interaction logged")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Team Optimization", "📊 Analytics", "🤖 AI Insights"])

with tab1:
    st.header("Team Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select Team Members")
        selected_people = []
        
        if st.session_state.pg.people:
            cols = st.columns(3)
            for idx, (name, person) in enumerate(st.session_state.pg.people.items()):
                with cols[idx % 3]:
                    label = f"{name} ({person.primary_type}"
                    if person.secondary_type:
                        label += f"/{person.secondary_type}"
                    label += ")"
                    if st.checkbox(label, key=f"sel_{name}"):
                        selected_people.append(person)
        else:
            st.warning("Add people first using the sidebar")
    
    with col2:
        st.subheader("Settings")
        team_size = st.slider("Team Size", 2, 10, 4)
        auto_optimize = st.checkbox("Auto-optimize selection", value=True)
    
    if len(selected_people) >= 2:
        st.markdown("---")
        
        if auto_optimize:
            optimal_team = build_optimal_team(st.session_state.pg, selected_people, team_size)
            st.subheader("🎯 Optimized Team")
        else:
            optimal_team = selected_people[:team_size]
            st.subheader("👥 Selected Team")
        
        # Display team
        team_cols = st.columns(min(len(optimal_team), 4))
        for idx, person in enumerate(optimal_team):
            with team_cols[idx % 4]:
                display = f"**{person.name}**\n\n{person.primary_type}"
                if person.secondary_type:
                    w = int(person.weights[0] * 100)
                    display += f"\n({w}%/{100-w}%)\n{person.secondary_type}"
                st.info(display)
        
        # Metrics
        cohesion = calc_team_cohesion(st.session_state.pg, optimal_team)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Team Cohesion", f"{cohesion:.3f}")
        with col2:
            mixed_count = sum(1 for p in optimal_team if p.secondary_type)
            st.metric("Mixed Personalities", mixed_count)
        with col3:
            unique_types = len(set(p.primary_type for p in optimal_team))
            st.metric("Unique Types", unique_types)
        
        # Compatibility matrix
        st.subheader("🔗 Compatibility Matrix")
        matrix_data = []
        for p1 in optimal_team:
            row = []
            for p2 in optimal_team:
                if p1 == p2:
                    row.append("-")
                else:
                    compat = st.session_state.pg.get_adjusted_compat(p1, p2)
                    row.append(f"{compat:.2f}")
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data, 
                         index=[p.name for p in optimal_team],
                         columns=[p.name for p in optimal_team])
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("📊 Organization Analytics")
    
    if st.session_state.pg.people:
        # Personality distribution
        st.subheader("Personality Distribution")
        primary_dist = {}
        for person in st.session_state.pg.people.values():
            primary_dist[person.primary_type] = primary_dist.get(person.primary_type, 0) + 1
        
        df_dist = pd.DataFrame(list(primary_dist.items()), columns=['Type', 'Count'])
        df_dist = df_dist.sort_values('Count', ascending=False)
        st.bar_chart(df_dist.set_index('Type'))
        
        # Interaction history
        if st.session_state.pg.interaction_history:
            st.subheader("Interaction History")
            df_hist = pd.DataFrame(st.session_state.pg.interaction_history)
            st.dataframe(df_hist, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Personnel", len(st.session_state.pg.people))
        with col2:
            mixed = sum(1 for p in st.session_state.pg.people.values() if p.secondary_type)
            st.metric("Mixed Personalities", mixed)
        with col3:
            st.metric("Interactions Logged", len(st.session_state.pg.interaction_history))
    else:
        st.info("No data yet. Add people to see analytics.")

with tab3:
    st.header("🤖 AI-Powered Insights")
    
    if len(selected_people) >= 2:
        if auto_optimize:
            team_to_analyze = optimal_team
        else:
            team_to_analyze = selected_people[:team_size]
        
        prompt, insights = generate_llm_insights(st.session_state.pg, team_to_analyze)
        
        st.subheader("Analysis Results")
        st.markdown(insights)
        
        # with st.expander("Show LLM Prompt & AI Thinking Process"):
        #     st.subheader("LLM Prompt")
        #     st.code(prompt, language='markdown')
        #     st.subheader("Raw AI Response")
        #     st.markdown(insights)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        # Find best additions
        available_others = [p for p in st.session_state.pg.people.values() 
                           if p not in team_to_analyze]
        
        if available_others:
            st.write("**Best candidates to add:**")
            scores = []
            for candidate in available_others:
                test_team = team_to_analyze + [candidate]
                new_cohesion = calc_team_cohesion(st.session_state.pg, test_team)
                current_cohesion = calc_team_cohesion(st.session_state.pg, team_to_analyze)
                improvement = new_cohesion - current_cohesion
                scores.append((candidate.name, candidate.primary_type, improvement))
            
            scores.sort(key=lambda x: x[2], reverse=True)
            for name, ptype, improvement in scores[:5]:
                if improvement > 0:
                    st.write(f"✅ {name} ({ptype}): +{improvement:.3f} cohesion improvement")
                else:
                    st.write(f"⚠️ {name} ({ptype}): {improvement:.3f} cohesion change")
    else:
        st.info("Select team members in the optimization tab to see AI insights")

st.markdown("---")
st.markdown("*Dynamic system with learning capabilities - tracks interactions and adapts recommendations*")