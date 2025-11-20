import json
import os
import typing

import transformers

class Text8Tokenizer(transformers.PreTrainedTokenizer):
  vocab_files_names = {'vocab_file': 'vocab.json'}

  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    vocab_file: typing.Optional[str] = None,
    **kwargs):
    if vocab_file is not None:
      with open(vocab_file, 'r', encoding='utf-8') as f:
        self._vocab_str_to_int = json.load(f)
    else:
      characters = list('abcdefghijklmnopqrstuvwxyz ')
      self._vocab_str_to_int = {
        '[CLS]': 0,
        '[SEP]': 1,
        '[BOS]': 2,
        '[EOS]': 3,
        '[MASK]': 4,
        '[PAD]': 5,
        '[RESERVED]': 6,
        '[UNK]': 7,
        ** {ch: i + 8 for i, ch in enumerate(characters)}}
    self.characters = [
      token for token in self._vocab_str_to_int.keys()
      if token not in {'[CLS]', '[SEP]', '[BOS]', '[EOS]', '[MASK]', '[PAD]',
                       '[RESERVED]', '[UNK]'}]
    self._vocab_int_to_str = {
      int(v): k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int

  def save_vocabulary(
    self,
    save_directory: str,
    filename_prefix: typing.Optional[str] = None,
  ) -> typing.Tuple[str]:
    os.makedirs(save_directory, exist_ok=True)
    vocab_filename = self.vocab_files_names['vocab_file']
    if filename_prefix:
      vocab_filename = f'{filename_prefix}-{vocab_filename}'
    vocab_path = os.path.join(save_directory, vocab_filename)
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
      json.dump(self._vocab_str_to_int, vocab_file, ensure_ascii=False)
    return (vocab_path,)
