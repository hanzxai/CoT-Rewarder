

import json, time, logging, requests

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(threadName)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

class APICallError(RuntimeError):
    """统一的失败异常，便于捕获根因"""
import re

# ---------- 正则：唯一且按顺序出现 <think> … </think> ----------
PATTERN = re.compile(
    r"""
    \A\s*
    <think>\s*(?P<think>.+?)\s*</think>   # 捕获 think
    \s*(?P<outside>.*?)\s*\Z              # 捕获 think 块外的全部文本（answer）
    """,
    re.DOTALL | re.VERBOSE,
)

def simple_tokenize(text: str) -> list[str]:
    """极简 tokenizer：按空白分割，过滤空串"""
    return [tok for tok in re.split(r"\s+", text) if tok]

# ---------- 主函数 ----------
def fine_format_reward(completions, **kwargs):
    rewards = []

    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue

        think = m.group("think").strip()
        # “think 块外”的内容视为 answer
        outside_text = (
            text[m.end("think") + len("</think>"):]
        )
        answer = outside_text.strip()

        reward = 0.0

        # ① think / answer 块都非空 → +0.1
        if think:
            reward += 0.1

            # ③ 长度比例 + outside > 100 → +0.5
            think_tokens = simple_tokenize(think)
            outside_tokens = simple_tokenize(outside_text)

            if len(outside_tokens) > 100 and len(think_tokens) >= 3 * len(outside_tokens):
                reward += 0.5

        # ② think 以指定句子结尾 → +0.4
        if think.rstrip().endswith("Everything seems correct."):
            reward += 0.4

        rewards.append(reward)

    return rewards

# ---------- 主函数 ----------
def fine_format_reward_v2(completions, **kwargs):
    rewards = []

    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue

        think = m.group("think").strip()
        # “think 块外”的内容视为 answer
        outside_text = (
            text[m.end("think") + len("</think>"):]
        )
        answer = outside_text.strip()

        reward = 0.0

        # ① think / answer 块都非空 → +0.1
        if think:
            reward += 0.1

            # ③ 长度比例 + outside > 100 → +0.5
            think_tokens = simple_tokenize(think)
            outside_tokens = simple_tokenize(outside_text)

            if len(outside_tokens) > 100 and len(think_tokens) >= 3 * len(outside_tokens):
                reward += 0.5

        # ② think 以指定句子结尾 → +0.4
        if 'correct' in think.split('/n')[-1]:
            reward += 0.05
        if 'Everything' in think.split('/n')[-1]:
            reward += 0.05
        if 'seems' in think.split('/n')[-1]:
            reward += 0.05
        if 'Everything seems correct.' in think.split('/n')[-1]:
            reward += 0.05
        if think.rstrip().endswith("Everything seems correct."):
            reward += 0.2

        rewards.append(reward)

    return rewards

def think_wait_reward(completions, least_number_waits=5,  **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue
        think = m.group("think").strip()
        # “think 块后”的内容视为 answer
        # outside_text = (
        #     text[m.end("think") + len("</think>"):]
        # )
        # answer = outside_text.strip()
        # 首先是wait和alternative累积reward
        # 然后是
        reward = 0.0
        think_tokens = simple_tokenize(think)
        for tt in think_tokens:
            if "wait" in tt:
                reward += 1.0/least_number_waits
                if reward >= 1.0:
                    reward = 1.0
                    break
        rewards.append(reward)
    return rewards

def think_alternative_reward(completions, least_number_alternatives=5,  **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue
        think = m.group("think").strip()
        reward = 0.0
        think_tokens = simple_tokenize(think)
        for tt in think_tokens:
            if "alternative" in tt:
                reward += 1.0/least_number_alternatives
                if reward >= 1.0:
                    reward = 1.0
                    break
        rewards.append(reward)
    return rewards

def think_verify_reward(completions, least_number_verifies=10,  **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue
        think = m.group("think").strip()
        reward = 0.0
        think_tokens = simple_tokenize(think)
        for tt in think_tokens:
            if "check" in tt or "verif" in tt or "justif" in tt or "reflect" in tt or 'confirm' in tt:
                reward += 1.0/least_number_verifies
                if reward >= 1.0:
                    reward = 1.0
                    break
        rewards.append(reward)
    return rewards

def call_api_4om(prompt, model='gpt-4o-mini', image_url=None,
                 max_retries=10, initial_retry_delay=5, request_id=None):
    request_id = request_id or f"{int(time.time()*1e3)}"
    for attempt in range(1, max_retries + 1):
        try:
            # 1. 构造 payload
            messages = (
                [{"role": "user",
                  "content": [{"type": "text", "text": prompt},
                              {"type": "image_url",
                               "image_url": {"url": image_url}}]}]
                if image_url else
                [{"role": "user", "content": prompt}]
            )
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.9,
                "max_completion_tokens": 32768,
                "top_p": 0.5,
                "repetition_penalty": 1.05,
                "user": "andy",
                "content_filter": False,
            }
            # 2. 调用
            resp = requests.post(URL, headers=HEADERS,
                                 json=payload, timeout=900)
            resp.raise_for_status()
            data = resp.json()

            # 3. 处理业务错误
            if 'error' in data:
                raise APICallError(f"API error: {data['error']}")

            content = data["choices"][0]["message"]["content"].strip()
            if not content:
               if not data["choices"][0]["message"]["reasoning_content"].strip():
                    raise APICallError("Empty completion")
               else:
                   content = data["choices"][0]["message"]["reasoning_content"].strip()

            try:
                score = float(content.split(r"\boxed{")[-1].split("}")[0])
            except Exception:
                raise APICallError(
                    f"Cannot parse score from:\n{content[-200:]}..."
                )

            if not (0 <= score <= 1):
                raise APICallError(f"Score out of range: {score}")

            # ✅ 成功
            return content

        except (requests.RequestException, APICallError) as e:
            if attempt == max_retries:
                log.error("[Req %s] failed permanently after %d tries – %s",
                          request_id, attempt, e)
                raise
            else:
                delay = initial_retry_delay * (2 ** (attempt - 1))
                log.warning(
                    "[Req %s] attempt %d/%d failed – %s → retry in %s s",
                    request_id, attempt, max_retries, e, delay
                )
                # if "Empty completion" in str(e) and  attempt >=4:
                #     # 我想获得completion一直为空的请求里的prompt
                #     log.error(f"Request {request_id} has empty completion, prompt: {prompt}")
                time.sleep(delay)
                # return None
    return None


def answer_final_reward(completions, **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue
        # think = m.group("think").strip()
        # “think 块后”的内容视为 answer
        outside_text = (
            text[m.end("think") + len("</think>"):]
        )
        answer = outside_text.strip()
        reward = 0.0
        try:
            if '/n/n' in answer:
                reward+=0.2
            if 'Final check' in answer:
                reward += 0.1
                if 'Final check' in answer.split('/n/n')[-2]:
                    reward += 0.1
                    if answer.split('/n/n')[-2].startwith('Final check'):
                        reward +=0.1
            if 'The final answer is ' in answer:
                reward += 0.1
                if 'The final answer is ' in answer.split('/n/n')[-1]:
                    reward += 0.1
                    if answer.split('/n/n')[-1].startswith('The final answer is '):
                        reward += 0.1
                        if answer.split('/n/n')[-1].startswith('The final answer is \\boxed{'):
                            reward += 0.2
        except Exception as e:
            reward=0.0
        rewards.append(reward)
    return rewards

def answer_final_enter1_reward(completions, **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        m = PATTERN.fullmatch(text)
        if not m:                      # 结构不合法 => 0
            rewards.append(0.0)
            continue
        # think = m.group("think").strip()
        # “think 块后”的内容视为 answer
        outside_text = (
            text[m.end("think") + len("</think>"):]
        )
        answer = outside_text.strip()
        reward = 0.0
        try:
            if '/n' in answer:
                reward+=0.2
            if 'Final check' in answer:
                reward += 0.2
                if 'Final check' in answer.split('/n')[-2]:
                    reward += 0.1
                    if answer.split('/n')[-2].startwith('Final check'):
                        reward +=0.1
            if 'The final answer is ' in answer:
                reward += 0.1
                if 'The final answer is ' in answer.split('/n')[-1]:
                    reward += 0.1
                    if answer.split('/n')[-1].startswith('The final answer is '):
                        reward += 0.1
                        if answer.split('/n')[-1].startswith('The final answer is \\boxed{'):
                            reward += 0.1
        except Exception as e:
            reward=0.0
        rewards.append(reward)
    return rewards



if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    from .utils.routed_sandbox import RoutedSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def has_wait_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = ['wait' in content for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def has_alternative_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = ['alternative' in content for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def get_exploring_reward(model='gpt-4o-mini'):

    def exploring_reward(completions, problem: list[str], **kwargs):
        rewards=[]
        num_completions = len(completions)
        for i in range(num_completions):
            completion = completions[i][0]["content"]
            prob = problem[i]
            om_return = call_api_4om(
                r'Score the following completion for thorough exploration, give your overall score from 0.0 to 1.0. A completion which consider have done thorough exploration includes using diverse strategies, explore all possibilities, use various methods to solve the problem, and performing check consistency of results across methods.  You can think step by step, and they end your generation with "The overall score is: \boxed{score}."\n\n' +'Problem:\n'+prob+'\n\nCompletion:' + completion,
                model=model)
            score_str = om_return.split(r"\boxed{")[-1].split("}")[0].strip()
            reward = float(score_str)
            rewards.append(reward)
        return rewards

    return exploring_reward


#--score_model google/gemini-2.5-pro-preview
def get_verifying_reward(model='gpt-4o-mini'):

    def verifying_reward(completions, problem: list[str], **kwargs):
        rewards=[]
        num_completions = len(completions)
        for i in range(num_completions):
            completion = completions[i][0]["content"]
            prob = problem[i]
            om_return = call_api_4om(
                r'Score the following completion for thorough verification, give your overall score from 0.0 to 1.0. A completion which consider have done verification includes actively performing rigorous verification and self-correction, giving justification for each step, actively reflect previous steps.  You can think step by step, and they end your generation with "The overall score is: \boxed{score}."\n\n' +'Problem:\n'+prob+'\n\nCompletion:' + completion,
                model=model)
            score_str = om_return.split(r"\boxed{")[-1].split("}")[0].strip()
            reward = float(score_str)
            rewards.append(reward)
        return rewards
    return verifying_reward

def get_decomposing_reward(model='gpt-4o-mini'):

    def decomposing_reward(completions, problem: list[str],  **kwargs):
        rewards=[]
        num_completions = len(completions)

        for i in range(num_completions):
            prob=problem[i]
            completion = completions[i][0]["content"]
            om_return = call_api_4om(
                r'Score the following completion for clear defining and systematically decomposing the problem in the beginning of the completion, give your overall score from 0.0 to 1.0. You can think step by step, and they end your generation with "The overall score is: \boxed{score}."\n\n' + 'Problem:\n' + prob + '\n\nCompletion:' + completion,
                model=model)
            score_str = om_return.split(r"\boxed{")[-1].split("}")[0].strip()
            reward= float(score_str)
            rewards.append(reward)
        return rewards
    return decomposing_reward


# def get_comprehensive_reward(model='gpt-4o-mini'):
#
#     def comprehensive_reward(completions, problem: list[str], **kwargs):
#         rewards=[]
#         num_completions = len(completions)
#
#         for i in range(num_completions):
#             prob = problem[i]
#             completion = completions[i][0]["content"]
#             om_return = call_api_4om(
#                 r'Score the following completion for the given problem, give your overall score from 0.0 to 1.0. You may consider many aspects, for example, how much this completion clear defines and systematically decomposes the problem in the beginning of the completion; how much it actively performs rigorous verification and self-correction, giving justification for each step, actively reflect previous steps; how much it has done thorough exploration includes using diverse strategies, explore all possibilities, use various methods to solve the problem, and performing check consistency of results across methods; beyond these common aspects, you may consider other aspects, e.g., problem-specific aspects. You can think step by step, and they end your generation with "The overall score is: \boxed{score}."\n\n' + 'Problem:\n' + prob + '\n\nCompletion:' + completion,
#                 model=model)
#             score_str = om_return.split(r"\boxed{")[-1].split("}")[0].strip()
#             reward= float(score_str)
#             rewards.append(reward)
#         return rewards
#     return comprehensive_reward
import threading
from typing import List, Dict

def get_comprehensive_reward(model: str = 'gpt-4o-mini'):
    """
    Returns a function that calculates comprehensive rewards for a list of completions.
    The reward calculation for each completion is done in a separate thread.
    """

    def comprehensive_reward(completions: List[List[Dict[str, str]]], problem: List[str], **kwargs) -> List[float]:
        """
        Calculates rewards for completions using multithreading.

        Args:
            completions: A list of completions. Each completion is a list containing a dictionary
                         with a "content" key.
            problem: A list of problem descriptions corresponding to each completion.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of float scores (rewards) for each completion.
        """
        rewards = [0.0] * len(completions)  # Pre-allocate list for results
        threads = []

        # Define a worker function to be executed by each thread
        def worker(index: int, prob_desc: str, completion_content: str):
            """Fetches and processes the reward for a single completion."""
            try:
                # Construct the prompt for the API call
                prompt = (
                    r'Score the following completion for the given problem, give your overall score from 0.0 to 1.0.  If the completion is empty, you can give a score of 0.'
                    r'You may consider many aspects, for example, how much this completion clearly defines and '
                    r'systematically decomposes the problem at the beginning of the completion; how much it actively '
                    r'performs rigorous verification and self-correction, giving justification for each step, '
                    r'actively reflect previous steps; how much it has done thorough exploration, including using '
                    r'diverse strategies, explore all possibilities, use various methods to solve the problem, '
                    r'and performing check consistency of results across methods; beyond these common aspects, '
                    r'you may consider other aspects, e.g., problem-specific aspects. You can think step by step, '
                    r'and they end your generation with "The overall score is: \boxed{score}."\n\n'
                    f'Problem:\n{prob_desc}\n\nCompletion:{completion_content}'
                )
                if len(completion_content)==0:
                    print('completion_content is empty')
                # else:
                #     print('completion_content is empty')
                om_return = call_api_4om(prompt, model=model)
                score_str = om_return.split(r"\boxed{")[-1].split("}")[0].strip()
                rewards[index] = float(score_str)
            except Exception as e:
                print(f"Error processing completion {index}: {e}")
                # Optionally, set a default error value or re-raise
                rewards[index] =0 # Or some other error indicator like -1.0 or None

        # Create and start a thread for each completion
        for i in range(len(completions)):
            prob = problem[i]
            # Assuming completions[i] is a list and we need the first item's "content"
            # e.g., completions = [[{"content": "solution1"}], [{"content": "solution2"}]]
            completion = completions[i][0]["content"]
            thread = threading.Thread(target=worker, args=(i, prob, completion))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return rewards

    return comprehensive_reward

def get_step_reward(model: str = 'gpt-4o-mini'):
    def step_reward(completions: List[List[Dict[str, str]]], problem: List[str], **kwargs):

        return
    return step_reward

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards

def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words
    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list
    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, e2b_router_url=e2b_router_url, **kwargs)
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output

def code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]
    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)

    if e2b_router_url is not None:
        routed_sandbox = RoutedSandbox(router_url=e2b_router_url)

        executions = routed_sandbox.run_code(
            scripts=scripts,
            language=language,
            timeout=30,
            request_timeout=28,
        )

        rewards = []
        for execution in executions:
            try:
                reward = float(execution.text)
                rewards.append(reward)
            except Exception:
                rewards.append(None)
        return rewards

    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        'wait': has_wait_reward, #
        'alternative': has_alternative_reward,#
        'think_wait': think_wait_reward,#
        'think_alternative': think_alternative_reward ,#
        'think_verify': think_verify_reward,#
        'answer_final':answer_final_reward,#
        'answer_final_enter1': answer_final_enter1_reward,#
        'fine_format':fine_format_reward,#
        'fine_format2':fine_format_reward_v2,#
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        'verifying': get_verifying_reward(script_args.score_model),#
        'decomposing': get_decomposing_reward(script_args.score_model),#
        'exploring': get_exploring_reward(script_args.score_model),#
        'comprehensive_reward': get_comprehensive_reward(script_args.score_model),#
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs